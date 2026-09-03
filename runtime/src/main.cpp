// main.cpp — runtime entrypoint.
//
//   CLI :  runtime --model m.gguf --prompt "Bună ziua" [--steps 24] [--target 256]
//   SERVER: runtime --model m.gguf --serve 8000
//
// The server exposes an OpenAI-compatible HTTP API (GET /health, POST /v1/completions,
// POST /v1/chat/completions) so it can be queried with curl or any OpenAI client — the
// same shape LM Studio uses. A request runs the masked-diffusion denoise loop over the
// model and returns the decoded text.
#include "diffusion/denoise.h"
#include "engine/model.h"
#include "tokenizer/tokenizer.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using dlm::BPETokenizer;
using dlm::RuntimeModel;
using dlm::discrete_generate_window;

// ---------------------------------------------------------------------------
// Minimal JSON (object / array / string / number / bool / null). No dependencies.
// ---------------------------------------------------------------------------
struct Json {
    enum Type { Null, Bool, Num, Str, Arr, Obj } type = Null;
    bool bfalse = false;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;

    static Json object() { Json j; j.type = Obj; return j; }
    static Json strv(std::string s) { Json j; j.type = Str; j.str = std::move(s); return j; }
    const Json* get(const std::string& k) const {
        for (auto& kv : obj) if (kv.first == k) return &kv.second;
        return nullptr;
    }
    void set(const std::string& k, Json v) { obj.emplace_back(k, std::move(v)); }
};

static std::string json_dump(const Json& j);
static Json json_parse(const std::string& s, size_t& i);

static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default: o += c;
        }
    }
    return o;
}

static std::string json_dump(const Json& j) {
    switch (j.type) {
        case Json::Null: return "null";
        case Json::Bool: return j.bfalse ? "false" : "true";
        case Json::Num: return std::to_string(j.num);
        case Json::Str: return "\"" + json_escape(j.str) + "\"";
        case Json::Arr: {
            std::string o = "[";
            for (size_t i = 0; i < j.arr.size(); ++i) { if (i) o += ","; o += json_dump(j.arr[i]); }
            return o + "]";
        }
        case Json::Obj: {
            std::string o = "{";
            for (size_t i = 0; i < j.obj.size(); ++i) {
                if (i) o += ",";
                o += "\"" + json_escape(j.obj[i].first) + "\":" + json_dump(j.obj[i].second);
            }
            return o + "}";
        }
    }
    return "null";
}

static void skip_ws(const std::string& s, size_t& i) { while (i < s.size() && isspace((unsigned char)s[i])) ++i; }

static Json json_parse(const std::string& s, size_t& i) {
    skip_ws(s, i);
    if (i >= s.size()) return Json();
    char c = s[i];
    if (c == '{') {
        Json j = Json::object();
        ++i; skip_ws(s, i);
        if (i < s.size() && s[i] == '}') { ++i; return j; }
        while (i < s.size()) {
            skip_ws(s, i);
            std::string key;
            if (s[i] == '"') { ++i; while (i < s.size() && s[i] != '"') { if (s[i]=='\\') ++i; key += s[i++]; } ++i; }
            skip_ws(s, i); ++i; // ':'
            j.set(key, json_parse(s, i));
            skip_ws(s, i);
            if (i < s.size() && s[i] == ',') { ++i; continue; }
            if (i < s.size() && s[i] == '}') { ++i; break; }
        }
        return j;
    }
    if (c == '[') {
        Json j; j.type = Json::Arr;
        ++i; skip_ws(s, i);
        if (i < s.size() && s[i] == ']') { ++i; return j; }
        while (i < s.size()) { j.arr.push_back(json_parse(s, i)); skip_ws(s, i); if (i < s.size() && s[i] == ',') { ++i; continue; } if (i < s.size() && s[i] == ']') { ++i; break; } }
        return j;
    }
    if (c == '"') {
        std::string o; ++i;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) { ++i; o += s[i++]; }
            else o += s[i++];
        }
        ++i;
        return Json::strv(o);
    }
    if (c == 't' && s.compare(i, 4, "true") == 0) { Json j; j.type = Json::Bool; i += 4; return j; }
    if (c == 'f' && s.compare(i, 5, "false") == 0) { Json j; j.type = Json::Bool; j.bfalse = true; i += 5; return j; }
    if (c == 'n' && s.compare(i, 4, "null") == 0) { i += 4; return Json(); }
    // number
    size_t start = i;
    while (i < s.size() && (isdigit((unsigned char)s[i]) || s[i] == '.' || s[i] == '-' || s[i] == 'e' || s[i] == 'E' || s[i] == '+')) ++i;
    Json j; j.type = Json::Num; j.num = std::stod(s.substr(start, i - start)); return j;
}

// ---------------------------------------------------------------------------
// Generation driver
// ---------------------------------------------------------------------------
struct GenParams {
    int steps = 24;
    int target = 256;
    int top_k = 200;
    float temperature = 0.0f;
    float top_p = 0.0f;
    float eps = 1e-3f;
    float alg_temp = 0.6f;
    std::string alg = "entropy";
    int block = 0;        // >0 => chained block-wise (long output)
    int max_new = 2048;
    uint64_t seed = 42;
};

static std::vector<int32_t> generate(
    RuntimeModel& model, const BPETokenizer& tok,
    const std::vector<int32_t>& prompt, const GenParams& p)
{
    auto forward = [&](const std::vector<int32_t>& ids) -> std::vector<float> {
        return model.forward(ids);
    };
    std::vector<int32_t> ids = prompt;
    if (p.block > 0) {
        int produced = 0;
        while (produced < p.max_new) {
            int n = std::min(p.block, p.max_new - produced);
            auto blk = discrete_generate_window(forward, model.mask_token_id(), ids,
                (int)ids.size() + n, p.steps, p.temperature, p.top_p, p.top_k,
                p.alg, p.eps, p.alg_temp, p.seed);
            for (int t : blk) ids.push_back((int32_t)t);
            produced += n;
        }
    } else {
        auto out = discrete_generate_window(forward, model.mask_token_id(), prompt,
            p.target, p.steps, p.temperature, p.top_p, p.top_k, p.alg, p.eps, p.alg_temp, p.seed);
        ids.insert(ids.end(), out.begin(), out.end());
    }
    return ids;
}

// ---------------------------------------------------------------------------
// HTTP server (OpenAI-compatible)
// ---------------------------------------------------------------------------
static std::atomic<bool> g_stop{false};
static void on_sig(int) { g_stop = true; }

static std::string serve_body(const std::string& req, RuntimeModel& model, const BPETokenizer& tok) {
    // route + JSON request
    std::string path;
    {
        size_t sp = req.find(' ');
        size_t sp2 = req.find(' ', sp + 1);
        if (sp != std::string::npos && sp2 != std::string::npos) path = req.substr(sp + 1, sp2 - sp - 1);
    }
    size_t hdr = req.find("\r\n\r\n");
    if (hdr == std::string::npos) return "{}";
    size_t cl = 0;
    {
        size_t p = req.find("Content-Length:");
        if (p != std::string::npos) cl = (size_t)std::strtol(req.c_str() + p + 15, nullptr, 10);
    }
    std::string body = req.substr(hdr + 4, std::min(cl, req.size() - hdr - 4));
    size_t bi = 0;
    Json j = json_parse(body, bi);

    // extract prompt (completions: "prompt", chat: last "messages[].content")
    std::string prompt;
    if (const Json* p = j.get("prompt")) { if (p->type == Json::Str) prompt = p->str; }
    else if (const Json* msgs = j.get("messages")) {
        for (auto& m : msgs->arr) {
            if (const Json* c = m.get("content")) if (c->type == Json::Str) prompt = c->str;
        }
    }

    GenParams gp;
    if (const Json* v = j.get("max_tokens")) {
        if (v->type == Json::Num && v->num > 0) gp.target = (int)v->num;
    }
    if (const Json* v = j.get("temperature")) gp.temperature = (float)v->num;

    std::vector<int32_t> ids = tok.encode(prompt);
    // hold prompt fixed, generate one window
    auto out = discrete_generate_window(
        [&](const std::vector<int32_t>& x) -> std::vector<float> { return model.forward(x); },
        model.mask_token_id(), ids, gp.target, gp.steps, gp.temperature, gp.top_p,
        gp.top_k, gp.alg, gp.eps, gp.alg_temp, gp.seed);
    std::string text = tok.decode(out);

    Json r = Json::object();
    Json choices = Json::object(); choices.type = Json::Arr;
    Json choice = Json::object();
    choice.set("text", Json::strv(text));
    choices.arr.push_back(choice);
    r.set("choices", choices);
    std::string resp = json_dump(r);
    std::string hdrout = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
        std::to_string(resp.size()) + "\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n";
    return hdrout + resp;
}

static void handle_conn(int fd, RuntimeModel& model, const BPETokenizer& tok) {
    std::string req;
    char buf[4096];
    ssize_t n;
    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        req.append(buf, (size_t)n);
        if (req.find("\r\n\r\n") != std::string::npos) break;
    }
    if (req.find("GET /health") == 0) {
        const char* h = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";
        write(fd, h, strlen(h));
    } else {
        std::string out = serve_body(req, model, tok);
        write(fd, out.data(), out.size());
    }
    close(fd);
}

static int serve(RuntimeModel& model, const BPETokenizer& tok, int port) {
    std::signal(SIGINT, on_sig);
    std::signal(SIGTERM, on_sig);
    int sfd = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{}; addr.sin_family = AF_INET; addr.sin_addr.s_addr = htonl(INADDR_ANY); addr.sin_port = htons((uint16_t)port);
    if (bind(sfd, (sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); return 1; }
    if (listen(sfd, 8) < 0) { perror("listen"); return 1; }
    std::printf("runtime server listening on http://localhost:%d  (openai-compatible)\n", port);
    std::thread th;
    while (!g_stop) {
        int fd = accept(sfd, nullptr, nullptr);
        if (fd < 0) break;
        th = std::thread(handle_conn, fd, std::ref(model), std::ref(tok));
        th.detach();
    }
    close(sfd);
    std::printf("server stopped\n");
    return 0;
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string model_path, prompt = "Bună ziua, aici este un test de difuzie.";
    GenParams gp;
    int serve_port = -1;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* { if (i + 1 < argc) return argv[++i]; std::fprintf(stderr, "%s needs a value\n", name); std::exit(1); };
        if (a == "--model") model_path = next(a.c_str());
        else if (a == "--prompt") prompt = next(a.c_str());
        else if (a == "--serve") serve_port = std::atoi(next(a.c_str()));
        else if (a == "--steps") gp.steps = std::atoi(next(a.c_str()));
        else if (a == "--target") gp.target = std::atoi(next(a.c_str()));
        else if (a == "--block") gp.block = std::atoi(next(a.c_str()));
        else if (a == "--max-new") gp.max_new = std::atoi(next(a.c_str()));
        else if (a == "--top-k") gp.top_k = std::atoi(next(a.c_str()));
        else if (a == "--temperature") gp.temperature = (float)std::atof(next(a.c_str()));
        else if (a == "--top-p") gp.top_p = (float)std::atof(next(a.c_str()));
        else if (a == "--alg") gp.alg = next(a.c_str());
        else if (a == "--alg-temp") gp.alg_temp = (float)std::atof(next(a.c_str()));
        else if (a == "--seed") gp.seed = std::strtoull(next(a.c_str()), nullptr, 10);
        else if (a == "--help") { std::printf("usage: runtime --model m.gguf [--prompt ...] [--serve PORT] [--steps N] [--target N] [--block N]\n"); return 0; }
        else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    if (model_path.empty()) { std::fprintf(stderr, "--model is required\n"); return 1; }

    RuntimeModel model(model_path);
    BPETokenizer tok;
    if (!tok.load(model_path.substr(0, model_path.find_last_of('/')))) {
        std::fprintf(stderr, "could not load tokenizer (vocab.json/merges.txt next to %s)\n", model_path.c_str());
        return 1;
    }

    if (serve_port > 0) return serve(model, tok, serve_port);

    auto ids = tok.encode(prompt);
    auto out_ids = generate(model, tok, ids, gp);
    std::string text = tok.decode(out_ids);
    std::printf("%s\n", text.c_str());
    return 0;
}
