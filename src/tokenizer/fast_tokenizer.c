#include "fast_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

static inline uint32_t hash_string(const char* str, size_t len) {
    uint32_t hash = 5381;
    for (size_t i = 0; i < len; i++) {
        hash = ((hash << 5) + hash) + (unsigned char)str[i];
    }
    return hash;
}

static int compare_vocab_entry(const void* a, const void* b) {
    return strcmp(*(const char**)a, *(const char**)b);
}

Tokenizer* tokenizer_load(const char* vocab_file) {
    int fd = open(vocab_file, O_RDONLY);
    if (fd < 0) {
        return NULL;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }

    void* data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    if (data == MAP_FAILED) {
        return NULL;
    }

    Tokenizer* tok = malloc(sizeof(Tokenizer));
    if (!tok) {
        munmap(data, st.st_size);
        return NULL;
    }

    tok->vocab_data = malloc(st.st_size + 1);
    if (!tok->vocab_data) {
        free(tok);
        munmap(data, st.st_size);
        return NULL;
    }
    
    memcpy(tok->vocab_data, data, st.st_size);
    tok->vocab_data[st.st_size] = '\0';
    tok->vocab_data_size = st.st_size;
    munmap(data, st.st_size);

    // count lines
    size_t line_count = 0;
    for (size_t i = 0; i < st.st_size; i++) {
        if (tok->vocab_data[i] == '\n') line_count++;
    }

    tok->vocab = malloc(line_count * sizeof(char*));
    tok->vocab_ids = malloc(line_count * sizeof(uint32_t));
    if (!tok->vocab || !tok->vocab_ids) {
        free(tok->vocab_data);
        free(tok->vocab);
        free(tok->vocab_ids);
        free(tok);
        return NULL;
    }

    // parse vocab
    size_t idx = 0;
    char* line = tok->vocab_data;
    for (size_t i = 0; i < st.st_size; i++) {
        if (tok->vocab_data[i] == '\n') {
            tok->vocab_data[i] = '\0';
            if (strlen(line) > 0) {
                tok->vocab[idx] = line;
                tok->vocab_ids[idx] = idx;
                idx++;
            }
            line = &tok->vocab_data[i + 1];
        }
    }
    tok->vocab_size = idx;

    // sort for binary search
    qsort(tok->vocab, tok->vocab_size, sizeof(char*), compare_vocab_entry);

    return tok;
}

void tokenizer_free(Tokenizer* tok) {
    if (!tok) return;
    free(tok->vocab_data);
    free(tok->vocab);
    free(tok->vocab_ids);
    free(tok);
}

static uint32_t find_token(Tokenizer* tok, const char* text, size_t len) {
    // binary search in sorted vocab
    size_t left = 0, right = tok->vocab_size;
    
    while (left < right) {
        size_t mid = (left + right) / 2;
        int cmp = strncmp(text, tok->vocab[mid], len);
        
        if (cmp == 0 && strlen(tok->vocab[mid]) == len) {
            return tok->vocab_ids[mid];
        } else if (cmp < 0) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    
    return 0;  // unknown token
}

TokenizedText* tokenizer_encode(Tokenizer* tok, const char* text) {
    if (!tok || !text) return NULL;

    TokenizedText* result = malloc(sizeof(TokenizedText));
    if (!result) return NULL;

    result->capacity = MAX_TOKENS;
    result->tokens = malloc(result->capacity * sizeof(uint32_t));
    if (!result->tokens) {
        free(result);
        return NULL;
    }
    result->length = 0;

    size_t text_len = strlen(text);
    size_t pos = 0;

    // greedy longest match tokenization
    while (pos < text_len) {
        if (result->length >= result->capacity) {
            result->capacity *= 2;
            uint32_t* new_tokens = realloc(result->tokens, result->capacity * sizeof(uint32_t));
            if (!new_tokens) {
                free(result->tokens);
                free(result);
                return NULL;
            }
            result->tokens = new_tokens;
        }

        // skip whitespace
        while (pos < text_len && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n')) {
            pos++;
        }
        
        if (pos >= text_len) break;

        // try longest match first
        size_t best_len = 0;
        uint32_t best_token = 0;
        
        for (size_t len = 1; len <= 50 && (pos + len) <= text_len; len++) {
            uint32_t token = find_token(tok, &text[pos], len);
            if (token != 0) {
                best_len = len;
                best_token = token;
            }
        }

        if (best_len > 0) {
            result->tokens[result->length++] = best_token;
            pos += best_len;
        } else {
            // unknown char, skip
            pos++;
        }
    }

    return result;
}

char* tokenizer_decode(Tokenizer* tok, const uint32_t* tokens, size_t length) {
    if (!tok || !tokens) return NULL;

    // estimate size
    size_t estimated_size = length * 10;  // rough estimate
    char* result = malloc(estimated_size);
    if (!result) return NULL;

    size_t pos = 0;
    for (size_t i = 0; i < length; i++) {
        if (tokens[i] >= tok->vocab_size) continue;
        
        const char* token_str = tok->vocab[tokens[i]];
        size_t token_len = strlen(token_str);
        
        if (pos + token_len + 2 > estimated_size) {
            estimated_size *= 2;
            char* new_result = realloc(result, estimated_size);
            if (!new_result) {
                free(result);
                return NULL;
            }
            result = new_result;
        }
        
        if (i > 0) result[pos++] = ' ';
        memcpy(&result[pos], token_str, token_len);
        pos += token_len;
    }
    
    result[pos] = '\0';
    return result;
}

void tokenized_text_free(TokenizedText* text) {
    if (!text) return;
    free(text->tokens);
    free(text);
}
