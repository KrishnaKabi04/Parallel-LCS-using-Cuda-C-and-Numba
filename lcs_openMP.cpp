#include <bits/stdc++.h>
#include<string>
#include<time.h>
#include "omp.h"
using namespace std;



int openMP_lcs(int **DP_matrix, string s1, string s2)
{

	// Loop is including s1.length because for lcs calc we use
	// 1 element more than the string size

	for(int i=0; i<=s1.length(); i++){
		DP_matrix[i][0] = 0;
	}

	// making elements 0 for 0 row and 0 column of matrix
	for(int j=0;j<=s2.length(); j++){
		DP_matrix[0][j] = 0;
	}

	int traverse_s1=1;

	for(int traverse_s2=1; traverse_s2<=s2.length(); traverse_s2++)
	{
		if(traverse_s1>s1.length())
			break;
		// number of anti diagnol elements to calculate
		int num_elem=min(traverse_s2,(int)s1.length()-traverse_s1);

		//parallelizing the for loop
		#pragma omp parallel for

		// calculate anti diagnol elements
		for(int k=0;k<=num_elem;k++)
		{
			int s1_a=traverse_s1+k, s2_b=traverse_s2-k;

			if (s1[s1_a-1]==s2[s2_b-1])
				DP_matrix[s1_a][s2_b]=DP_matrix[s1_a-1][s2_b-1]+1;
			else
				DP_matrix[s1_a][s2_b]=max(DP_matrix[s1_a-1][s2_b], DP_matrix[s1_a][s2_b-1]);

		}

		if(traverse_s2>=s2.length())
		{
			traverse_s2=s2.length()-1, traverse_s1++;
		}

	}

	return DP_matrix[s1.length()][s2.length()];
}

int main(int argc, char *argv[])
{
	FILE *fp;

	int len_S1, len_S2;
	int **DP_matrix;

	fp = fopen(argv[1],"r");
	fscanf(fp, "%d %d", &len_S1, &len_S2);

	char* s1 = new char[len_S1];
	char* s2 = new char[len_S2];

	fscanf(fp, "%s %s", s1, s2);

	cout << "String 1: "<< s1 << endl;
	cout << "String 2: "<< s2 << endl;

	// added 1 because for LCS matrix length is 1 more than size of string
	DP_matrix = new int*[len_S1+1];
	for(int i=0;i<len_S1+1;i++)
	{
		DP_matrix[i] = new int[len_S2+1];
	}

	double t = omp_get_wtime();

	int length_lcs = openMP_lcs(DP_matrix,s1,s2);
	cout << "Longest Substring Length is:" << length_lcs << endl;
	cout<< omp_get_wtime()-t << "seconds - Time taken" << endl;
}
