import time
import argparse
from numba import cuda, jit
import math
import numpy as np 

@cuda.jit
def Find_L_entry(mat_d: np.ndarray, seqA :  np.ndarray, seqA_len: int, seqB:  np.ndarray, seqB_len: int, diag: int):
    
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
    i = diag - j    

    #print("j: ", j ,"i: ", i)
    if (i >= 0 and i < seqB_len):
        #print("test: ",seqA[i],seqB[j])
        #print("inside if: i: ", i ,"j: ", j)
        #print("second", seqA[j], seqB[i])
        if seqA[j]==seqB[i]:
            mat_d[i+1][j+1]= mat_d[i][j]+1
            #print("if val: ", mat_d[i+1][j+1])
        else:
            mat_d[i+1][j+1]= max(mat_d[i][j+1], mat_d[i+1][j])
            #print("else val: ", mat_d[i+1][j+1], "max: ", mat_d[i][j+1],mat_d[i+1][j] )
    


#using DP method CPU side
def validate_LCS(seqA, seqB, DP):

    for i in range(0, len(seqB)):  # 1 to n-1
        for j in range(0, len(seqA)):
            if seqA[j] == seqB[i]:
                DP[i+1, j+1] = DP[i, j] + 1
            else:
                DP[i+1, j+1] = max(DP[i, j+1], DP[i+1, j])
    return DP
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-A", "--seqA", help="input file name")
    parser.add_argument("-B", "--seqB", help="input file name")
    parser.add_argument("-TB", "--TB", help="input thread block")
    parser.add_argument("-V", "--valid", default= False, help="validation with CPU DP algorithm")
    parser.add_argument("-V_only", "--only_valid", default=False, help="only validation without running in GPU")
    
    args = parser.parse_args()
    
    with open(args.seqA) as f:
            content= f.readlines()
        
    seqA= content[0]
    seqA_len= len(seqA)
    #assert seqA_len==len(seqA)
    
    with open(args.seqB) as f:
        content= f.readlines()
        
    seqB= content[0]
    seqB_len= len(seqB)
    #assert seqB_len==len(seqB)
    
    if seqB_len > seqA_len:
        seqA, seqB= seqB, seqA
        seqA_len, seqB_len= seqB_len,seqA_len
        
    char_map={'A':65, 'C':67, 'G':71, 'T':84}
    #convert to array
    seqA_arr= np.array([*map(char_map.get, seqA)])
    seqB_arr= np.array([*map(char_map.get, seqB)])
    
    print("Length of A: {}, {}".format(len(seqA), seqA_len))
    
    #assert seqA_len==len(seqA_arr)
    #assert seqB_len==len(seqB_arr)
    
    #print("seq A: ", seqA)
    #print("seq B: ", seqB)
    
    #print("seq A array: ", seqA_arr)
    #print("seq B array: ", seqB_arr)
        
    if seqA_len < 2 ^ 8:
        mat_h = np.zeros((seqB_len+1, seqA_len+1), dtype=np.uint8)
        #DP = np.zeros((seqB_len, seqA_len), dtype=np.uint8)
        
    elif seqA_len < 2 ^ 16:
        mat_h = np.zeros((seqB_len+1, seqA_len+1), dtype=np.uint16)
        #DP = np.zeros((seqB_len, seqA_len), dtype=np.uint8)
        
    else:
        mat_h = np.zeros((seqB_len+1, seqA_len+1), dtype=np.uint32)
        #DP = np.zeros((seqB_len, seqA_len), dtype=np.uint8)
    
    DP= mat_h #for CPU side validation
    
    if args.only_valid==False:
    
        gpu=cuda.get_current_device()
        print("gpu.name: ", gpu.name)
        
        d_A = cuda.to_device(seqA_arr)
        d_B = cuda.to_device(seqB_arr)
        mat_d = cuda.to_device(mat_h)
        cuda.synchronize()
    
        max_diag_len= seqA_len if seqA_len > seqB_len else seqB_len
        diag_count = seqA_len + seqB_len - 1
        print("max_diag_len: ",max_diag_len, "diag_count: ",diag_count)
    
        #//set up blocks
        threadsperblock = int(args.TB)
        blockspergrid = math.ceil( (max_diag_len-1)/threadsperblock )
        print("blockspergrid: ",blockspergrid)
        
        #Invoke kernel
        start = time.time()
        for diag in range(0,diag_count):
            #print("\n ----------------- Iteration: {} -------------------- \n".format(diag+1))
            #print(seqA_arr[1], seqB_arr[1])
            #print(type(seqA_arr))
            Find_L_entry[blockspergrid,threadsperblock](mat_d, d_A, seqA_len, d_B, seqB_len, diag)
            cuda.synchronize() #check ----
        end = time.time()
        print("Time for computing S in GPU with compilation is: {:.2f} secs".format(end-start))
        
        #for cached version
        #start = time.time()
        #for diag in range(0,diag_count):
        #    #print("\n ----------------- Iteration: {} -------------------- \n".format(diag+1))
        #    #print(seqA_arr[1], seqB_arr[1])
        #    #print(type(seqA_arr))
        #    Find_L_entry[blockspergrid,threadsperblock](mat_d, d_A, seqA_len, d_B, seqB_len, diag)
        #    cuda.synchronize() #check ----
        #end = time.time()
        #print("Time for computing S in GPU after compilation is: {:.2f} secs".format(end-start))
        
        mat_h = mat_d.copy_to_host()
        cuda.synchronize()
        
        #print("mat_h shape: ", mat_h.shape)
        #print("Matrix h: \n", mat_h)
    
        if args.valid==True:
            try:
                start = time.time()
                DP= validate_LCS(seqA, seqB, DP)
                end = time.time()
                #print("DP: ", DP)
                assert DP[-1][-1]==mat_h[-1][-1]
                print("Time for computing LCS in CPU using DP is: {:.2f} secs".format(end - start))
                print("\n\n Validation True! LCS length is: {} \n\n".format(mat_h[-1][-1]))
            except Exception as e:
                print(e.__class__, "LCS don't match! GPU: ", mat_h[-1][-1], " CPU: ",DP[-1][-1])
            
    else:
        print("Only running on CPU")
        start = time.time()
        DP= validate_LCS(seqA, seqB, DP)
        end = time.time()
        print("\nTime for computing LCS in CPU using DP is: {:.2f} secs".format(end - start))
        print("\n\nLCS length on CPU is: {} \n\n".format(DP[-1][-1]))
    