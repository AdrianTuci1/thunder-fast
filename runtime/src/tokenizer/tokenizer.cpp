// tokenizer/tokenizer.cpp — byte-level BPE implementation (Qwen2 / GPT-2 style).
#include "tokenizer/tokenizer.h"

#include <fstream>
#include <sstream>

namespace dlm {

bool BPETokenizer::load(const std::string& dir) {
    // vocab.json
    {
        std::ifstream f(dir + "/vocab.json");
        if (!f) return false;
        std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        size_t i = 0;
        while (i < s.size()) {
            size_t q = s.find('"', i);
            if (q == std::string::npos) break;
            size_t q2 = s.find('"', q + 1);
            std::string tok = s.substr(q + 1, q2 - q - 1);
            size_t colon = s.find(':', q2);
            size_t comma = s.find(',', colon);
            int id = std::stoi(s.substr(colon + 1, (comma == std::string::npos ? s.size() : comma) - colon - 1));
            if ((size_t)id >= id_to_token_.size()) id_to_token_.resize((size_t)id + 1);
            id_to_token_[(size_t)id] = tok;
            token_to_id_[tok] = id;
            i = (comma == std::string::npos) ? s.size() : comma + 1;
        }
    }
    // merges.txt
    {
        std::ifstream f(dir + "/merges.txt");
        if (!f) return false;
        std::string line;
        int rank = 0;
        std::getline(f, line); // skip "#version: ..." header
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            merge_rank_[line.substr(0, sp) + " " + line.substr(sp + 1)] = rank++;
        }
    }
    return true;
}

int32_t BPETokenizer::token_to_id(const std::string& tok) const {
    auto it = token_to_id_.find(tok);
    return it == token_to_id_.end() ? -1 : it->second;
}

int32_t BPETokenizer::add_token(const std::string& tok, int32_t id) {
    id_to_token_.resize((size_t)id + 1);
    id_to_token_[(size_t)id] = tok;
    token_to_id_[tok] = id;
    return id;
}

const std::string& BPETokenizer::id_to_token(int32_t id) const { return id_to_token_[(size_t)id]; }

std::vector<int32_t> BPETokenizer::encode(const std::string& text) const {
    std::vector<int32_t> out;
    std::istringstream words(text);
    std::string word;
    while (words >> word) {
        // byte-level: operate on raw bytes, then BPE-merge the byte sequence.
        std::vector<std::string> syms;
        for (unsigned char b : word) syms.emplace_back(1, (char)b);
        while (syms.size() > 1) {
            int best_rank = 1 << 30;
            size_t best_i = 0;
            bool found = false;
            for (size_t i = 0; i + 1 < syms.size(); ++i) {
                auto it = merge_rank_.find(syms[i] + " " + syms[i + 1]);
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second; best_i = i; found = true;
                }
            }
            if (!found) break;
            syms[best_i] = syms[best_i] + syms[best_i + 1];
            syms.erase(syms.begin() + (long)best_i + 1);
        }
        for (const auto& s : syms) {
            int id = token_to_id(s);
            out.push_back(id >= 0 ? id : 0); // unknown -> fallback token
        }
    }
    return out;
}

std::string BPETokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int id : ids) out += id_to_token_[(size_t)id];
    return out;
}

} // namespace dlm
