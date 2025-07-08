#pragma once
#include <omp.h>
#include "Python.h"

#ifdef OMP_ACCELERATOR
__attribute__((constructor))
static void init_omp() {
	if (omp_get_max_threads() == 1)
    	omp_set_num_threads(6);
}
#define OMP_U_ _Pragma ("omp parallel for")
#define OMP_U_2 _Pragma ("omp parallel for collapse(2)")
#define OMP_U_3 _Pragma ("omp parallel for collapse(3)")
#else
#define OMP_U_
#define OMP_U_2
#define OMP_U_3
#endif

void INIT_THREADS(int num) {
	omp_set_num_threads(num);
}

typedef struct Matrix {
    int row, col;
    float** val;
} Matrix;
typedef struct Tensor {
	Matrix** mat;
	int depth;
} Tensor;

Matrix* new_matrix(int row, int col) {
    Matrix* newm = (Matrix*)malloc(sizeof(Matrix));
	newm->val = (float**)malloc(row* sizeof(float*));
	for (int i = 0; i < row; i++) newm->val[i] = (float*)calloc(col, sizeof(float));
	newm->row = row, newm->col = col;
	return newm;
}
void copy_matrix(Matrix* dest, Matrix* sauce) {
	if (!sauce) return ;
	OMP_U_2
    for (int i = 0; i < sauce->row; i++) {
        for (int j = 0; j < sauce->col; j++) dest->val[i][j] = sauce->val[i][j];
    }
}
Matrix* get_copy_matrix(Matrix* sauce) {
	if (!sauce) return NULL;
	Matrix* newm = new_matrix(sauce->row, sauce->col);
	copy_matrix(newm, sauce);
	return newm;
}
void fill_matrix(Matrix* m, float fill_value) {
	OMP_U_2
	for (int i = 0; i < m->row; i++) {
		for (int j = 0; j < m->col; j++) m->val[i][j] = fill_value;
	}
}
Matrix* summ(Matrix* a, Matrix* b) {
    if (!a || !b || a->row != b->row || a->col != b->col) {
        printf("Error: Inappropriate type of matrixes for calculating sum !!");
        return NULL;
    }
    Matrix* newm = new_matrix(a->row, b->col);
	OMP_U_2
	for (int i = 0; i < a->row; i++) {
		for (int j = 0; j < b->col; j++) newm->val[i][j] = a->val[i][j] + b->val[i][j];
	}
    return newm;
}
void get_summ(Matrix* dest, Matrix* sauce) {
	if (!sauce) return ;
    if (dest->row != sauce->row || dest->col != sauce->col) {
        printf("Error: Inappropriate type of matrixes for calculating sum !!");
        return ;
    }
	OMP_U_2
    for (int i = 0; i < sauce->row; i++) {
		for (int j = 0; j < sauce->col; j++) dest->val[i][j] += sauce->val[i][j];
	}
}
Matrix* minusm(Matrix* a, Matrix* b) {
    if (!a || !b || a->row != b->row || a->col != b->col) {
        printf("Error: Inappropriate type of matrixes for calculating minus !!");
        return NULL;
    }
    Matrix* newm = new_matrix(a->row, b->col);
	OMP_U_2
    for (int i = 0; i < a->row; i++) {
		for (int j = 0; j < b->col; j++) newm->val[i][j] = a->val[i][j] - b->val[i][j];
	}
    return newm;
}
void get_minusm(Matrix* dest, Matrix* sauce) {
	if (!sauce) return ;
    if (dest->row != sauce->row || dest->col != sauce->col) {
        printf("Error: Inappropriate type of matrixes for calculating minus !!");
        return ;
    }
	OMP_U_2
    for (int i = 0; i < sauce->row; i++) {
		for (int j = 0; j < sauce->col; j++) dest->val[i][j] -= sauce->val[i][j];
	}
}
Matrix* multiply(Matrix* a, Matrix* b) {
    if (!a || !b || a->col != b->row) {
        printf("Error: Inappropriate type of matrixes for multiplying !!");
        return NULL;
    }
    Matrix* newm = new_matrix(a->row, b->col);
	OMP_U_2
    for (int i = 0; i < a->row; i++) {
		for (int j = 0; j < b->col; j++) {
			float sum = 0;
			for (int k = 0; k < a->col; k++) sum += a->val[i][k]* b->val[k][j];
			newm->val[i][j] = sum;
		}
	}
    return newm;
}
void get_multiply(Matrix* multiplication, Matrix* a, Matrix* b) {
	if (!a || !b || a->col != b->row || multiplication->row < a->row || multiplication->col < b->col) {
        printf("Error: Inappropriate type of matrixes for multiplying !!");
        return ;
    }
	OMP_U_2
	for (int i = 0; i < a->row; i++) {
		for (int j = 0; j < b->col; j++) {
			float sum = 0;
			for (int k = 0; k < a->col; k++) sum += a->val[i][k]* b->val[k][j];
			multiplication->val[i][j] = sum;
		}
	}
}
void scalar_multiply(Matrix* a, float n) {
	OMP_U_2
	for (int i = 0; i < a->row; i++) {
		for (int j = 0; j < a->col; j++) a->val[i][j] *= n;
	}
}
Matrix* transpose(Matrix* m) {
	if (!m) return NULL;
	Matrix* newm = new_matrix(m->col, m->row);
	OMP_U_2
	for (int i = 0; i < m->col; i++) {
		for (int j = 0; j < m->row; j++) newm->val[i][j] = m->val[j][i];
	}
	return newm;
}
Matrix* ewise_multiply(Matrix* a, Matrix* b) {
    if (!a || !b || a->row != b->row || a->col != b->col) {
        printf("Error: Inappropriate type of matrixes for ew multiplying !!");
        return NULL;
    }
    Matrix* newm = new_matrix(a->row, a->col);
	OMP_U_2
    for (int i = 0; i < a->col; i++) {
		for (int j = 0; j < a->row; j++) newm->val[j][i] = a->val[j][i]* b->val[j][i];
	}
	return newm;
}
void get_ewise_multiply(Matrix* dest, Matrix* sauce) {
	if (!sauce) return ;
    if (dest->row != sauce->row || dest->col != sauce->col) {
        printf("Error: Inappropriate type of matrixes for ew multiplying !!");
        return ;
    }
	OMP_U_2
    for (int i = 0; i < sauce->row; i++) {
		for (int j = 0; j < sauce->col; j++) dest->val[i][j] *= sauce->val[i][j];
	}
}
void get_rot_180m(Matrix* dest, Matrix* sauce) {
	if (!sauce) return ;
    if (dest->row != sauce->row || dest->col != sauce->col) {
        printf("Error: Inappropriate type of matrixes for rotating 180 degree !!");
        return ;
    }
	OMP_U_2
	for (int i = 0; i < sauce->row; i++) {
		for (int j = 0; j < sauce->col; j++) dest->val[i][j] = sauce->val[sauce->row - i - 1][sauce->col - j - 1];
	}
}
Matrix* rot_180m(Matrix* m) {
	Matrix* rot = new_matrix(m->row, m->col);
	get_rot_180m(rot, m);
	return rot;
}
void print_matrix(Matrix* m, int decimal) {
    int i, j;
	for (i = 0; i < m->row; i++) {
		if (i != 0) printf("\n [");
        else printf("[[");
		printf("%.*f", decimal, m->val[i][0]);
		for (j = 1; j < m->col; j++) printf(", %.*f", decimal, m->val[i][j]);
		printf("]");
	}
	printf("]\n");
}
void free_matrix(Matrix* m) {
    if (!m) return ;
    if (m->val) {
        for (int i = 0; i < m->row; i++) if (m->val[i]) free(m->val[i]);
        free(m->val);
    }
    free(m);
}

Tensor* new_tensor(int row, int col, int depth) {
	Tensor* newt = (Tensor*)malloc(sizeof(Tensor));
	newt->depth = depth;
	newt->mat = (Matrix**)malloc(depth* sizeof(Matrix*));
	for (int i = 0; i < depth; i++) newt->mat[i] = new_matrix(row, col);
	return newt;
}
void copy_tensor(Tensor* dest, Tensor* sauce) {
	if (dest->depth < sauce->depth) {
		printf("Warning: Inappropriate size of tensors for copying !!");
		return ;
	}
	for (int i = 0; i < sauce->depth; i++) copy_matrix(dest->mat[i], sauce->mat[i]);
}
Tensor* get_copy_tensor(Tensor* sauce) {
	if (!sauce) return NULL;
	Tensor* newt = new_tensor(sauce->mat[0]->row, sauce->mat[0]->col, sauce->depth);
	copy_tensor(newt, sauce);
	return newt;
}
void fill_tensor(Tensor* t, float fill_value) {
	for (int i = 0; i < t->depth; i++) fill_matrix(t->mat[i], fill_value);
}
void print_tensor(Tensor* t, int decimal) {
	for (int i = 0; i < t->depth; i++) print_matrix(t->mat[i], decimal);
}
void free_tensor(Tensor* t) {
	if (!t) return ;
	if (t->mat) {
		for (int i = 0; i < t->depth; i++) free_matrix(t->mat[i]);
	}
	free(t);
}
Tensor* sumt(Tensor* a, Tensor* b) {
	if (!a || !b || a->depth != b->depth) {
		printf("Error: Inappropriate size of tensors for calculating sum !!");
		return NULL;
	}
	Tensor* sum = (Tensor*)malloc(sizeof(Tensor));
	sum->depth = a->depth;
	sum->mat = (Matrix**)malloc(sum->depth* sizeof(Matrix));
	for (int i = 0; i < sum->depth; i++) sum->mat[i] = summ(a->mat[i], b->mat[i]);
	return sum;
}
void get_sumt(Tensor* dest, Tensor* sauce) {
	if (!sauce) return ;
	if (dest->depth != sauce->depth) {
		printf("Error: Inappropriate size of tensors for calculating sum !!");
		return ;
	}
	for (int i = 0; i < sauce->depth; i++) get_summ(dest->mat[i], sauce->mat[i]);
}
Tensor* minust(Tensor* a, Tensor* b) {
	if (!a || !b || a->depth != b->depth) {
		printf("Error: Inappropriate size of tensors for calculating minus !!");
		return NULL;
	}
	Tensor* min = (Tensor*)malloc(sizeof(Tensor));
	min->depth = a->depth;
	min->mat = (Matrix**)malloc(min->depth* sizeof(Matrix));
	for (int i = 0; i < min->depth; i++) min->mat[i] = minusm(a->mat[i], b->mat[i]);
	return min;
}
void get_minust(Tensor* dest, Tensor* sauce) {
	if (!sauce) return ;
	if (dest->depth != sauce->depth) {
		printf("Error: Inappropriate size of tensors for calculating minus !!");
		return ;
	}
	for (int i = 0; i < sauce->depth; i++) get_minusm(dest->mat[i], sauce->mat[i]);
}
void get_tensor_ewise_multiply(Tensor* dest, Tensor* sauce) {
	if (!sauce) return ;
    if (dest->depth != sauce->depth) {
        printf("Error: Inappropriate size of tensors for ew multiplying !!");
        return ;
    }
	for (int i = 0; i < sauce->depth; i++) get_ewise_multiply(dest->mat[i], sauce->mat[i]);
}
Tensor* tensor_ewise_multiply(Tensor* a, Tensor* b) {
	if (!a || !b) return NULL;
	if (a->depth != b->depth) {
        printf("Error: Inappropriate size of tensors for ew multiplying !!");
        return NULL;
    }
	Tensor* t = (Tensor*)malloc(sizeof(Tensor));
	t->depth = a->depth;
	t->mat = (Matrix**)malloc(t->depth* sizeof(Matrix*));
	for (int i = 0; i < t->depth; i++) t->mat[i] = ewise_multiply(a->mat[i], b->mat[i]);
	return t;
}
void tensor_scalar_multiply(Tensor* a, float n) {
	if (!a) return ;
	for (int i = 0; i < a->depth; i++) scalar_multiply(a->mat[i], n);
}

void copy_vector(float* dest, float* sauce, int length) {
	OMP_U_
	for (int i = 0; i < length; i++) dest[i] = sauce[i];
}
float* get_copy_vector(const float* sauce, int length) {
	float* newv = (float*)malloc(length* sizeof(float));
	OMP_U_
	for (int i = 0; i < length; i++) newv[i] = sauce[i];
	return newv;
}
float* sumv(float* a, float* b, int length) {
	if (!a || !b) return NULL;
	float* newv = (float*)malloc(length* sizeof(float));
	OMP_U_
	for (int i = 0; i < length; i++) newv[i] = a[i] + b[i];
	return newv;
}
void get_sumv(float* dest, float* sauce, int length) {
	if (!dest || !sauce) return ;
	OMP_U_
	for (int i = 0; i < length; i++) dest[i] += sauce[i];
}
float* minusv(float* a, float* b, int length) {
	if (!a || !b) return NULL;
	float* newv = (float*)malloc(length* sizeof(float));
	OMP_U_
	for (int i = 0; i < length; i++) newv[i] = a[i] - b[i];
	return newv;
}
void get_minusv(float* dest, float* sauce, int length) {
	if (!dest || !sauce) return ;
	OMP_U_
	for (int i = 0; i < length; i++) dest[i] -= sauce[i];
}
void vector_scale(float* v, float n, int length) {
	if (!v) return ;
	OMP_U_
	for (int i = 0; i < length; i++) v[i] *= n;
}
void vector_ewise(float* dest, float* sauce, int length) {
	if (!dest || !sauce) return ;
	OMP_U_
	for (int i = 0; i < length; i++) dest[i] *= sauce[i];
}

float mean(float* y, int length) {
	float total = 0.0;
	OMP_U_
	for (int i = 0; i < length; i++) total += y[i];
	return total / length;
}
float median(float* s, int length) {
	if (length == 1) return s[0];
	float median;
	int as = 0, is = 0;
	float* max_heap = (float*)calloc((length / 2 + 2), sizeof(float));
	float* min_heap = (float*)calloc((length / 2 + 2), sizeof(float));
	max_heap[as++] = s[0];
	for (int i = 1; i < length; i++) {
		if (as == is) {
			if (s[i] > min_heap[0]) {
				heap_add(max_heap, &as, min_heap[0], 1);
				heap_remove(min_heap, &is, min_heap[0], 0);
				heap_add(min_heap, &is, s[i], 0);
			} else heap_add(max_heap, &as, s[i], 1);
		} else {  // as > is
			if (s[i] < max_heap[0]) {
				heap_add(min_heap, &is, max_heap[0], 0);
				heap_remove(max_heap, &as, max_heap[0], 1);
				heap_add(max_heap, &as, s[i], 1);
			} else heap_add(min_heap, &is, s[i], 0);
		}
	}
	if (as == is) median = (max_heap[0] + min_heap[0]) / 2;
	else median = max_heap[0];
	free(max_heap);
	free(min_heap);
	return median;
}
float sum_square_error(float* y_pred, float* y_true, int length) {
	float sum = 0.0;
	OMP_U_
	for (int i = 0; i < length; i++) sum += (y_pred[i] - y_true[i])*(y_pred[i] - y_true[i]);
	return sum;
}
float mean_square_error(float* y_pred, float* y_true, int length) {
	return sum_square_error(y_pred, y_true, length) / length;
}
