#ifndef FAST_TOKENIZER_H
#define FAST_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#define MAX_TOKENS 512
#define MAX_VOCAB_SIZE 50000

typedef struct {
    char** vocab;
    uint32_t* vocab_ids;
    size_t vocab_size;
    char* vocab_data;
    size_t vocab_data_size;
} Tokenizer;

typedef struct {
    uint32_t* tokens;
    size_t length;
    size_t capacity;
} TokenizedText;

Tokenizer* tokenizer_load(const char* vocab_file);
void tokenizer_free(Tokenizer* tok);

TokenizedText* tokenizer_encode(Tokenizer* tok, const char* text);
char* tokenizer_decode(Tokenizer* tok, const uint32_t* tokens, size_t length);
void tokenized_text_free(TokenizedText* text);

#endif
