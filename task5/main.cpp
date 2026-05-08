#include "mpi.h"
#include <stdio.h>

#define N 8
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int A[N][N], B[N][N];
  int len[N], disp[N];
  MPI_Datatype diag_type;
  MPI_Status status;

  printf("A:\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i][j] = 10 * i + j;
      B[i][j] = 0;
      printf("%02d\t", A[i][j]);
    }
    printf("\n");
  }

  for (int i = 0; i < N; i++) {
    len[i] = 1;
    disp[i] = i * N + i;
  }
  MPI_Type_indexed(N, len, disp, MPI_INT, &diag_type);
  MPI_Type_commit(&diag_type);
  
  MPI_Sendrecv(&A, 1, diag_type, 0, 111, &B, 1, diag_type, 0, 111,
               MPI_COMM_SELF, &status);

  printf("B:\n");
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++)
      printf("%02d\t", B[i][j]);
    printf("\n");
  }
  MPI_Type_free(&diag_type);

  MPI_Finalize();
  return 0;
}
