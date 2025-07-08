#pragma once
#include "D:\Std_gcc_&_g++_plus\Python.h"
#define Nan (0.0f/0.0f)

typedef struct Node {
	int data;
	struct Node* next;
} Node;
typedef struct FNode {
	float data;
	struct FNode* next;
} FNode;
typedef struct Space {
    int key;
	char* keys;
	struct Space* next;
} Space;
typedef struct Set {
    Space** arr;
	unsigned max_size;
	int size;
} Set;

Node* new_node(int val) {
	Node* newn = (Node*)malloc(sizeof(Node));
	newn->data = val;
	newn->next = NULL;
	return newn;
}
void add_node(Node** head, int val) {
	if (*head == NULL) {
		*head = new_node(val);
		return ;
	}
	Node* cur = *head;
	while (cur->next != NULL) cur = cur->next;
	cur->next = new_node(val);
}
void free_llist(Node** head) {
	Node* cur = *head;
	while (*head != NULL) {
		*head = (*head)->next;
		free(cur);
		cur = *head;
	}
}
FNode* new_fnode(float val) {
	FNode* newn = (FNode*)malloc(sizeof(FNode));
	newn->data = val;
	newn->next = NULL;
	return newn;
}
void add_fnode(FNode** head, float val) {
	if (*head == NULL) {
		*head = new_fnode(val);
		return ;
	}
	FNode* cur = *head;
	while (cur->next != NULL) cur = cur->next;
	cur->next = new_fnode(val);
}
void free_fllist(FNode** head) {
	FNode* cur = *head;
	while (*head != NULL) {
		*head = (*head)->next;
		free(cur);
		cur = *head;
	}
}
void swap(float* a, float* b) {
	float t = *a;
	*a = *b;
	*b = t;
}
void heapify_tree(float* s, int size, int i, int is_max_heap) {  // heapify down
	int cur = i, left = 2*i + 1, right = 2*i + 2;
	if (size == 1 || size == 0) {
		return ;
	} else if (is_max_heap == 1) {
		if (left < size && s[left] > s[cur]) cur = left;
		if (right < size && s[right] > s[cur]) cur = right;
	} else if (is_max_heap == 0) {
		if (left < size && s[left] < s[cur]) cur = left;
		if (right < size && s[right] < s[cur]) cur = right;
	}
	if (cur != i) {
		swap(&s[i], &s[cur]);
		heapify_tree(s, size, cur, is_max_heap);
	}
}
void heap_add(float* s, int* size, int val, int is_max_heap) {
    int i = (*size)++, parent;
    s[i] = val;
    while (i > 0) {
        parent = (i - 1) / 2;
        if ((is_max_heap && s[i] > s[parent]) || (!is_max_heap && s[i] < s[parent])) {  // heapify up
            swap(&s[i], &s[parent]);
            i = parent;
        } else break;
    }
}
void heap_remove(float* s, int* size, int val, int is_max_heap) {
    int i;
    for (i = 0; i < *size && val != s[i]; i++) ;
	if (i == *size) return ;
    swap(&s[i], &s[--(*size)]);
    heapify_tree(s, *size, i, is_max_heap);
}
int is_prime(int n) {
	if(n < 2) return 0;
    for(int i = 2; i*i <= n; i++){
        if(n % i == 0) {
            return 0;
        }
    }
    return 1;
}
void trans_prime(int* n) {
	if (*n % 2 == 0) (*n)++;
	while (!is_prime(*n)) *n += 2;
}
int hash_func(const char* keys, int max_size) {
    if (!keys) return 0;
	int hash = 0;
	while (*keys) hash = (hash* 31) + (*keys++);
	return hash % max_size;
}
Set* new_set(int max_size) {
	Set* st = (Set*)malloc(sizeof(Set));
	trans_prime(&max_size);
	st->max_size = max_size;
	st->size = 0;
	st->arr = (Space**)malloc(st->max_size* sizeof(Space*));
	for (int i = 0; i < st->max_size; i++) {
		st->arr[i] = NULL;
	}
	return st;
}
void set_add(Set* st, const char* keys, int key) {
	int id = hash_func(keys, st->max_size);
	Space* head = st->arr[id];
	Space* mid = head;
	while (mid != NULL) {
		if (!strcmp(mid->keys, keys)) return ;
		mid = mid->next;
	}
	Space* news = (Space*)malloc(sizeof(Space));
	news->keys = strdup(keys);
    news->key = key;
	news->next = head;
	st->arr[id] = news;
	st->size++;
}
int set_find(Set* st, const char* keys) {
    int id = hash_func(keys, st->max_size);
    Space* mid = st->arr[id];
    while (mid != NULL) {
        if (!strcmp(mid->keys, keys)) return 1;
        mid = mid->next;
	}
    return 0;
}
int set_call(Set* st, const char* keys) {
    int id = hash_func(keys, st->max_size);
    Space* mid = st->arr[id];
    while (mid != NULL) {
        if (!strcmp(mid->keys, keys)) return mid->key;
        mid = mid->next;
	}
    return -1;
}
int* set_key_access(Set* st, const char* keys) {
    int id = hash_func(keys, st->max_size);
    Space* mid = st->arr[id];
    while (mid != NULL) {
        if (!strcmp(mid->keys, keys)) return &(mid->key);
        mid = mid->next;
	}
    return NULL;
}
void free_set(Set* st) {
	if (!st) return ;
    Space* mid, * next;
	for (int i = 0; i < st->max_size; i++) {
        mid = st->arr[i];
        while (mid != NULL) {
            next = mid->next;
            if (mid->keys) free(mid->keys);
            free(mid);
            mid = next;
        }
    }
    free(st->arr);
    free(st);
}