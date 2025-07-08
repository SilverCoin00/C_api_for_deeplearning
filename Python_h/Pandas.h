#pragma once
#include "Python.h"

typedef struct Data_Frame {
    char** features;
    float** data;
	char*** str_data;
	int* str_cols;
    int row, col;          // row, col of float data
} Data_Frame;

static int is_blank(char* s) {
	for (int i = 0; s[i]; i++) if (s[i] != ' ' && s[i] != '\t' && s[i] != '\n' && s[i] != '\r') return 0;;
	return 1;
}
static int is_nan(char* s) {
	if (!s) return 0;
	s[strcspn(s, "\n")] = '\0';
	const char* nan[] = {"NA", "NAN", "NaN", "N/A", "nan", "null", "NULL", "None", "missing", "?", "--", "-", ""};
	for (int i = 0; i < 13; i++) {
		if (!strcmp(s, nan[i])) return 1;
	}
	return 0;
}
static int count_row(FILE* file, int max_line_length) {
	char* s = (char*)malloc(max_line_length* sizeof(char));
	int count;
	for (count = 0; fgets(s, max_line_length, file); ) if (!is_blank(s)) count++;;
	free(s);
	rewind(file);
	return count;
}
static int count_col(FILE* file, int max_line_length, char* seperate) {
	char* s = (char*)malloc(max_line_length* sizeof(char));
	if (fgets(s, max_line_length, file) == NULL) {
		free(s);
		return 0;
	}
	int count = 1;
	for (int i = 0; s[i] != '\n'; i++) {
		if (s[i] == seperate[0]) count++;
	}
	free(s);
	rewind(file);
	return count;
}
static void find_string_col(FILE* file, int max_line_length, char* seperate, int** str_cols) {
	char* s = (char*)malloc(max_line_length* sizeof(char));
	if (fgets(s, max_line_length, file) == NULL) {
		free(s);
		*str_cols = (int*)malloc(sizeof(int));
		(*str_cols)[0] = 0;
		return ;
	}
	fgets(s, max_line_length, file);    // constrained first data line is the standard
	Node* list_char = NULL;
	int count = 0;
	for (int i = 0, chk = 0, col = 0; s[i]; i++) {
		if (((s[i] >= 65 && s[i] <= 90) || (s[i] >= 97 && s[i] <= 122)) && chk == 0 && s[i] != seperate[0]) {
			count++;
			chk = 1;
			add_node(&list_char, col);
		}
		if (s[i] == seperate[0]) {
			chk = 0;
			col++;
		}
	}
	*str_cols = (int*)malloc((count + 1)* sizeof(int));
	(*str_cols)[0] = count;
	Node* mid = list_char;
	for (int i = 0; i < count; i++, mid = mid->next) (*str_cols)[i + 1] = mid->data;
	free(s);
	free_llist(&list_char);
	rewind(file);
}
int strtoi(const char* number) {
	int num = 0;
	for (int i = 0; number[i]; i++) {
		if (number[i] < 48 || number[i] > 57) return -1;
		num *= 10;
		num += number[i] - 48;
	}
	return num;
}
void itostr(int num, char* s) {
	int i = num, j;
	if (num == 0) {
		s[0] = '0';
		s[1] = '\0';
		return ;
	}
	for (j = 0; i > 0; i /= 10, j++);
	s[j] = '\0';
	for (j-- ; j >= 0; num /= 10, j--) s[j] = num % 10 + '0';
}
void ftostr(float num, char* s, int num_nums) {
	int intg, i, j;
	for (j = 0; num >= 1.0; num /= 10, j++);
	for (i = 0, num *= 10 ; i < num_nums; i++) {
		if (i == j) {
			s[i] = '.';
			num_nums++;
			continue;
		}
		intg = (int) num;
		s[i] = intg % 10 + '0';
		num *= 10;
	}
	s[i] = '\0';
}
static int is_data_line(char* s, char* seperate) {
	int i;
	for (i = 0; s[i] != seperate[0] && s[i] != '.'; i++) ;
	char* copy = (char*)malloc((i + 1)* sizeof(char));
	strncpy(copy, s, i);
	copy[i] = '\0';
	if (strcmp(copy, "-1") && strtoi(copy) == -1) {
		free(copy);
		return 1;
	}
	free(copy);
	return 0;
}
Data_Frame* read_csv(char* file_name, int max_line_length, char* seperate) {
    FILE* file = fopen(file_name, "r");
	if (!file) {
		printf("Error: Cannot open file !!");
		return NULL;
	}
	int i, j, k, size;
    Data_Frame* newd = (Data_Frame*)malloc(sizeof(Data_Frame));
	newd->row = count_row(file, max_line_length) - 1;
	find_string_col(file, max_line_length, seperate, &(newd->str_cols));
    newd->col = count_col(file, max_line_length, seperate) - newd->str_cols[0];
	newd->features = (char**)malloc(newd->col* sizeof(char*));
	newd->data = (float**)malloc(newd->row* sizeof(float*));
	if (newd->str_cols[0] != 0) newd->str_data = (char***)malloc(newd->row* sizeof(char**));

    char* s = (char*)malloc(max_line_length* sizeof(char));
	char* token;
    fgets(s, max_line_length, file);
    if (is_data_line(s, seperate)) {
		s[strcspn(s, "\n")] = '\0';
		token = strtok(s, seperate);
		j = newd->col + newd->str_cols[0];
		for (i = 0; token != NULL && i < j; i++) {
			size = strlen(token) + 1;
			newd->features[i] = (char*)malloc(size* sizeof(char));
			snprintf(newd->features[i], size, "%s", token);
			token = strtok(NULL, seperate);
		}
	} else rewind(file);

	float num;
	for (i = 0, j = 0; i < newd->row; i++, j = 0) {
		newd->data[i] = (float*)malloc(newd->col* sizeof(float));
		if (newd->str_cols[0] != 0) newd->str_data[i] = (char**)malloc(newd->str_cols[0]* sizeof(char*));
        fgets(s, max_line_length, file);
		s[strcspn(s, "\n")] = '\0';
		token = strtok(s, seperate);
		for (k = 0; token != NULL && j < newd->col + newd->str_cols[0]; ) {
			size = (k < newd->str_cols[0]) ? newd->str_cols[k + 1] : -1;
			if (j != size) {
				if (is_nan(token)) num = 0.0f / 0.0f;
				else sscanf(token, "%f", &num);
				newd->data[i][j++] = num;
			} else {
				size = strlen(token);
				newd->str_data[i][k] = (char*)malloc((size >= 3 ? size : 3) + 1);
				if (!is_nan(token)) strcpy(newd->str_data[i][k++], token);
				else strcpy(newd->str_data[i][k++], "nan");
			}
			token = strtok(NULL, seperate);
		}
	}
    free(s);
    fclose(file);
    return newd;
}
void make_csv(char* file_name_csv, Data_Frame* df, char* seperate) {
    FILE* file = fopen(file_name_csv, "w");
	if (!file) {
		printf("Error: Cannot open file !!");
		return ;
	}
	if (!df) return ;
	int i, j;
	if (df->features) {
		for (i = 0; i < df->col; i++) 
			fprintf(file, "%s%s", df->features[i] ? df->features[i] : "null", i == df->col - 1 ? "\n" : seperate);
	}
	for (i = 0; i < df->row; i++) {
		for (j = 0; j < df->col; j++) {
			fprintf(file, "%f%s", df->data[i][j], j == df->col - 1 ? "\n" : seperate);
		}
	}
	fclose(file);
}
void print_data_frame(Data_Frame* df, int col_space, int num_of_rows) {
    if (!df) {
        printf("Error: DataFrame is not existing !!");
        return ;
    }
	if (num_of_rows < 0 || num_of_rows > df->row) num_of_rows = df->row;
    int i, j, k, cur;
	if (df->features) {
		printf("\t");
		j = df->col + df->str_cols[0];
		for (i = 0; i < j; i++) printf("%*s ", col_space, df->features[i]);
		printf("\n");
	}
    for (i = 0; i < num_of_rows; i++) {
        printf("%5d\t", i + 1);
        for (j = 0, k = 0; j < df->col || k < df->str_cols[0]; ) {
			cur = (k < df->str_cols[0]) ? df->str_cols[k + 1] : -1;
			if (j != cur) {
				if (df->data[i][j] == df->data[i][j]) printf("%*.2f ", col_space, df->data[i][j]);
				else printf("%*s ", col_space, "nan");
				j++;
			}
			else printf("%*s ", col_space, df->str_data[i][k++]);
		}
        printf("\n");
    }
}
void free_data_frame(Data_Frame* df) {
	if (!df) return ;
	int i, j;
    if (df->features) {
		for (i = 0; i < df->col; i++) 
			if (df->features[i]) free(df->features[i]);
		free(df->features);
	}
	if (df->data) {
		for (i = 0; i < df->row; i++) 
			if (df->data[i]) free(df->data[i]);
		free(df->data);
	}
	if (df->str_cols[0] > 0) {
		for (i = 0; i < df->row; i++) {
			for (j = 0; j < df->str_cols[0]; j++) free(df->str_data[i][j]);
			free(df->str_data[i]);
		}
		free(df->str_data);
	}
	free(df->str_cols);
    free(df);
}
void describe_df_digit(Data_Frame* df, int col_space) {
	if (!df) {
        printf("Error: DataFrame is not existing !!");
        return ;
    }
    int i, j, k;
	if (df->features) {
		printf("\t");
		for (i = 0, k = 0; i < df->col; i++, k++) {
			for (j = 1; j <= df->str_cols[0]; j++) 
				if (k == df->str_cols[j]) k++;
			printf("%*s ", col_space, df->features[k]);
		}
		printf("\n");
	}
	float** descb = (float**)malloc(df->col* sizeof(float*));
	for (i = 0; i < df->col; i++) {
		descb[i] = (float*)calloc(5, sizeof(float));
		descb[i][3] = 1e10;
		descb[i][4] = 1e-10;
		for (j = 0; j < df->row; j++) {
			if (df->data[j][i] == df->data[j][i]) {
				descb[i][0]++;
				descb[i][1] += df->data[j][i];
				if (descb[i][3] > df->data[j][i]) descb[i][3] = df->data[j][i];
				if (descb[i][4] < df->data[j][i]) descb[i][4] = df->data[j][i];
			}
		}
		descb[i][1] /= descb[i][0];
		for (j = 0; j < df->row; j++)
			if (df->data[j][i] == df->data[j][i])
				descb[i][2] += (df->data[j][i] - descb[i][1])* (df->data[j][i] - descb[i][1]);
		descb[i][2] /= (descb[i][0] - 1);
		descb[i][2] = (float) sqrt((float) descb[i][2]);
	}
	printf("count\t");
	for (i = 0; i < df->col; i++) printf("%*d ", col_space, (int)descb[i][0]);
	printf("\n mean\t");
	for (i = 0; i < df->col; i++) printf("%*.4f ", col_space, descb[i][1]);
	printf("\n std \t");
	for (i = 0; i < df->col; i++) printf("%*.4f ", col_space, descb[i][2]);
	printf("\n min \t");
	for (i = 0; i < df->col; i++) printf("%*.4f ", col_space, descb[i][3]);
	printf("\n max \t");
	for (i = 0; i < df->col; i++) {
		printf("%*.4f ", col_space, descb[i][4]);
		free(descb[i]);
	}
	printf("\n");
	free(descb);
}
void describe_df_string(Data_Frame* df, int col_space) {
	if (df->str_cols[0] == 0) {
		printf("DataFrame has no string data !\n");
		return ;
	}
	int i, j, k;
	float** descb = (float**)malloc(df->str_cols[0]* sizeof(float*));
	char** cdescb = (char**)malloc(df->str_cols[0]* sizeof(char*));
	Set** str_frequence = (Set**)malloc(df->str_cols[0]* sizeof(Set*));
	for (i = 0; i < df->str_cols[0]; i++) {
		descb[i] = (float*)calloc(3, sizeof(float));
		str_frequence[i] = new_set(17);
		for (j = 0; j < df->row; j++) {
			if (strcmp(df->str_data[j][i], "nan")) {
				descb[i][0]++;
				if (set_find(str_frequence[i], df->str_data[j][i]))
					(*set_key_access(str_frequence[i], df->str_data[j][i]))++;
				else set_add(str_frequence[i], df->str_data[j][i], 1);
			}
		}
		descb[i][1] = str_frequence[i]->size;
	}
	Space* mid;
	for (k = 0; k < df->str_cols[0]; k++) {
		for (i = 0, j = 0; i < str_frequence[k]->max_size; i++) {
			mid = str_frequence[k]->arr[i];
			while (mid != NULL) {
				if (mid->key > j) {
					j = strlen(mid->keys) + 1;
					cdescb[k] = (char*)malloc(j* sizeof(char));
					strcpy(cdescb[k], mid->keys);
					j = mid->key;
					descb[k][2] = j;
				}
				mid = mid->next;
			}
		}
		free_set(str_frequence[k]);
	}
	free(str_frequence);
	if (df->features) {
		printf("\t");
		for (i = 1; i <= df->str_cols[0]; i++)
			printf("%*s ", col_space, df->features[df->str_cols[i]]);
		printf("\n");
	}
	printf(" count\t");
	for (i = 0 ; i < df->str_cols[0]; i++) printf("%*d ", col_space, (int) descb[i][0]);
	printf("\nunique\t");
	for (i = 0 ; i < df->str_cols[0]; i++) printf("%*d ", col_space, (int) descb[i][1]);
	printf("\n  top \t");
	for (i = 0 ; i < df->str_cols[0]; i++) {
		printf("%*s ", col_space, cdescb[i]);
		free(cdescb[i]);
	}
	printf("\n  fre \t");
	for (i = 0 ; i < df->str_cols[0]; i++) {
		printf("%*d ", col_space, (int) descb[i][2]);
		free(descb[i]);
	}
	printf("\n");
	free(descb);
	free(cdescb);
}
