#include <bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void print_matched_sequence(int* big_L, int n, int m, int lcs_max);     

/* Kernel to find L entry */

__global__ void LCS_Kernel(int *S_d, int *seq1_d, int m,int *seq2_d, int n, int lcs_diag_max, int diag) {

    int j = threadIdx.x + blockIdx.x*blockDim.x;
    int i = diag - j;
    int ind_iseqj, ind_ijseq;
    int max_seq = lcs_diag_max + 1;
    int* L = &S_d[(n+1)*max_seq + max_seq]; 

    if (i >= 0 && i < n){
          if (seq1_d[i] == seq2_d[j]) {
           L[i*(n+1)+j] = L[(i-1)*(n+1)+(j-1)] + 1;   
        } else {
            if (i == 0) ind_iseqj = 0;
            else ind_iseqj = L[(i-1)*(n+1)+j];
            if (j == 0) ind_ijseq = 0;
            else ind_ijseq = L[i*(n+1)+(j-1)];
            if (ind_iseqj >= ind_ijseq) {
                L[i*(n+1)+j] = ind_iseqj;  
            } else {
                L[i*(n+1)+j] = ind_ijseq;
            }
        }
        }
}  /* LCS_Kernel */

static int BLOCK_SIZE = 8;

/* Host code */

int main(int argc, char* argv[]) {

    FILE *fp;
    int len_S1, len_S2;
	
    fp = fopen(argv[1], "r");
    fscanf(fp, "%d %d", &len_S1, &len_S2);

    char* s1 = new char[len_S1];
    char* s2 = new char[len_S2];

    fscanf(fp, "%s %s", s1, s2);
  

    int lcs_max;
    int lcs_diag, lcs_diag_max, diag;
    int* S_size_h;

    /* device pointers */
    int* seq1_d;
    int* seq2_d;
    int* S_size_d;

    /* host Pointers */
    int* seq1_h = new int[len_S1];
    int* seq2_h = new int[len_S2];


     for (int i = 0; i < len_S1; i++) {
        seq1_h[i] = s1[i] - 'A' + 1;
     }

     for (int i = 0; i < len_S2; i++) {
        seq2_h[i] = s2[i] - 'A' + 1;
     }

   //printing sequences

  /* printf("Sequence A: ");

    for(int i=0; i< len_S1; i++){
       printf("%d   ", seq1_h[i]);
     }
     printf("\n");

     printf("Sequence B: ");

     for(int i=0; i< len_S2; i++){
        printf("%d     ", seq2_h[i]);
     }

     printf("\n");
*/
    if (len_S1 < len_S2)
       lcs_diag_max = lcs_max = len_S1;
    else
       lcs_diag_max = lcs_max = len_S2;


    lcs_diag = len_S1 + len_S2 - 1;
    unsigned long long int S_size = (len_S1+1)*(len_S2+1)*(lcs_max+1);
    S_size_h = (int*) calloc(S_size, sizeof(int));
    printf("S_size: %d \n", S_size);

    /* Allocate seq1, seq2 & big_L in device memory */

    cudaMalloc((void**)&seq1_d, len_S1*sizeof(int));
    cudaMalloc((void**)&seq2_d, len_S2*sizeof(int));
    cudaMalloc((void**)&S_size_d, S_size*sizeof(int));

    cudaDeviceSynchronize();

   /* Copy vectors from host memory to device memory */

    cudaMemcpy(seq1_d, seq1_h, len_S1*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(seq2_d, seq2_h, len_S2*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(S_size_d, S_size_h, S_size*sizeof(int),cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

        //set up blocks

    dim3 DimGrid( (lcs_diag_max-1)/BLOCK_SIZE +1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    //printf("DimGrid: %d, DimBlock: %d \n", ((lcs_diag_max-1)/BLOCK_SIZE +1), DimBlock);

    struct timeval start, end;

    gettimeofday(&start, NULL);
    for (diag = 0; diag < lcs_diag; diag++) {
        //printf("\n ----------------- Iteration: %d -------------------- \n", diag+1);
        LCS_Kernel<<<DimGrid,DimBlock>>>(S_size_d, seq1_d, len_S1,seq2_d, len_S2, lcs_diag_max, diag);
        cudaDeviceSynchronize();
    }
    
    time_taken = ((float) ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1.0e6));
    printf("time taken = %f \n", time_taken);

    cudaMemcpy(S_size_h, S_size_d, S_size*sizeof(int), cudaMemcpyDeviceToHost);

    //print_matched_sequence(S_size_h, len_S2, len_S1, lcs_max); //, finish-start);
	
    /* Free device memory */
    cudaFree(seq1_d);
    cudaFree(seq2_d);
    cudaFree(S_size_d);

    /* Free host memory */
    free(seq1_h);
    free(seq2_h);
    free(S_size_h);

    return 0;
}   /* main */

void print_matched_sequence(int* L_arr, int n,int m, int lcs_max){
	
    int i, j, L_size, p;
    int* L;

    printf("Inside print results....\n");

    L_size = (n+1)*(m+1);
    L = &L_arr[1*(n+1)*(lcs_max+1) + 1*(lcs_max+1)];

    printf("test\n\n\n\n");
    printf("L =\n");

    for (i = 0; i < L_size; i+=m) {
               printf("i: %d  ",i);
        for (j = i; j < i+m; j++){
                        printf("%d ", L[j]);
                }
        printf("\n");
    printf("\n");
    }

    printf("size of L: %d \n",sizeof(L));

    p = L[(n-1)*(n+1)*(lcs_max+1) + (n-1)*(lcs_max+1)];

    printf("The longest common subsequence length: %d \n", p);               
}













