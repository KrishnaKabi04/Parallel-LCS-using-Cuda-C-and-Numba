import time
import argparse
from numba import cuda, jit
import multiprocessing as mp
from utils.gpu import get_Smat
import math
import numpy as np 
import logging
from datetime import datetime

#LOG_DIR= "../parallel_LCS/logs/"

@cuda.jit
def get_Smat(row: int, d_P: np.ndarray, S: np.ndarray):

    #print("test")
    #print("cuda.threadIdx.x: ", cuda.threadIdx.x)
    #print(" cuda.blockDim.x: ", cuda.blockDim.x)
    #print("cuda.blockIdx.x: ",cuda.blockIdx.x)
    
    j = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # same as cuda.grid(1)
    #print("j: ",j)
    
    #Equation 6 in PDF
    while j < S.shape[1]:
        t = d_P[j] != 0
        s = (0 - S[row - 1, j] + t * S[row - 1, d_P[j] - 1]) != 0
        S[row, j] = S[row - 1, j] + t * (s ^ 1)
        j += cuda.blockDim.x * cuda.gridDim.x
        
def init_S(n: int, m: int):
    if m < 2 ^ 8:
        S = np.zeros((n, m), dtype=np.uint8)
    elif m < 2 ^ 16:
        S = np.zeros((n, m), dtype=np.uint16)
    else:
        S = np.zeros((n, m), dtype=np.uint32)
    return S

#using DP method CPU side
def validate_LCS(seqA, seqB, S):
    for i in range(1, len(seqA)):  # 1 to n-1
        for j in range(1, len(seqB)):
            if seqA[i] == seqB[j]:
                S[i, j] = S[i-1, j-1] + 1
            else:
                S[i, j] = max(S[i-1, j], S[i, j-1])
    return S

def _get_Pi(seqB: str, seqB_len:int, letter: str) -> dict:
    Pi = {letter: []} #dictionary of list of letters and their increasing occurence
    
    for j in range(seqB_len):
        if j == 0:
            Pi[letter].append(0)
        elif seqB[j] == letter:
            Pi[letter].append(j)
        else:
            Pi[letter].append(Pi[letter][-1])

    if seqB_len < 255:
        Pi[letter] = np.array(Pi[letter], dtype=np.uint8)
    elif seqB_len < 65535:
        Pi[letter] = np.array(Pi[letter], dtype=np.uint16)
    else:
        Pi[letter] = np.array(Pi[letter], dtype=np.uint32)
    return Pi
    
def find_P(seqB: str, seqB_len:int):
    start = time.time()
    
    #multiprocessing on CPU side
    pool = mp.Pool(processes=4) #for 4 characters
    processes = pool.starmap_async(_get_Pi, [(seqB, seqB_len, letter) for letter in "ATCG"])
    Pi_list = processes.get()
    
    #for sequential
    #Pi_list=[]
    #for letter in "ATCG":
    #    Pi_list.append(_get_Pi(seqB, seqB_len, letter))
    
    #print(type(Pi_list))
    #print(Pi_list)

    #P = {**Pi_list[0], **Pi_list[1], **Pi_list[2], **Pi_list[3]}
    P = np.array([Pi_list[0]['A'], Pi_list[1]['T'], Pi_list[2]['C'], Pi_list[3]['G']])
    #print(type(P))
    #print("P: ", P)
    end = time.time()
    t = end - start
    print("Time for computing P is: {:.2f} s".format(t))
    return P, t
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-A", "--seqA", help="input file name")
    parser.add_argument("-B", "--seqB", help="input file name")
    parser.add_argument("-TB", "--TB", help="input thread block")
    parser.add_argument("-V", "--valid", default= True, help="validation with CPU DP algorithm")
    #parser.add_argument("-V_only", "--only_valid", default=False, help="only validation without running in GPU")
        
    args = parser.parse_args()
    
    gpu=cuda.get_current_device()
    print("gpu.name: ", gpu.name)
    
    with open(args.seqA) as f:
        content= f.readlines()
    
    seqA= "0"+content[0]
    seqA_len= len(seqA)
    #assert seqA_len==len(seqA)
    
    with open(args.seqB) as f:
        content= f.readlines()
        
    seqB= "0" +content[0]
    seqB_len= len(seqB)
    #assert seqB_len==len(seqB)

    if seqB_len > seqA_len:
        seqA, seqB= seqB, seqA
        seqA_len, seqB_len= seqB_len,seqA_len
    
    #logging
    #logging.basicConfig(level=logging.INFO, format='%(message)s')
    #logger = logging.getLogger(__name__)
    #print('logger: ', logger)
    #time_stmp= datetime.now().strftime("%m%d%Y-%H%M%S")
    #logger.addHandler(logging.FileHandler(LOG_DIR+'optimized_'+str(seqA_len)+"_"+time_stmp+'.log', 'a'))
    #print = logger.info
    
    print("SeqA length: {}, SeqB length: {}".format(seqA_len, seqB_len))
    #print("SeqA: {} ".format(seqA))
    #print("SeqB: {} ".format(seqB))
    
    time_elap = []

    # Initialize S
    S = init_S(seqA_len, seqB_len)
    DP= S #for CPU validation
    print("S shape: {}".format(S.shape))
    
    P, t = find_P(seqB, seqB_len)
    #print("P[0]:", P[0])
    #P_new= np.array(P[0], P[1], P[2], P[3])
    time_elap.append(t)
    #print("P_new: ", P.shape)
    
    print("Allocating GPU memory")
    d_S = cuda.to_device(S)
    d_P=  cuda.to_device(P)
    cuda.synchronize()
    print("copied data to GPU")
    
    #print("P: ",P)
    print("Length of A: {}, {}".format(len(seqA), seqA_len))
    #print("A:",seqA)
    
    threadsperblock = int(args.TB)
    blockspergrid = int(math.ceil(seqA_len / threadsperblock))
    print("blockspergrid: {}".format(blockspergrid))
    
    char_map= {'A':0, 'T':1, 'C': 2, 'G':3}
    
    #Numba has to compile your function for the argument types given before it executes the machine code version of your function, this takes time. 
    #However, once the compilation has taken place Numba caches the machine code version of your function for the particular types of arguments presented. 
    #If it is called again the with same types, it can reuse the cached version instead of having to compile again.
    
    start = time.time()
    for i in range(1, seqA_len):  # 1 to n-1  Length of Seq A
        #print("-------------------- Iteration {} --------------------------".format(i))
        ai = seqA[i]  #each charof Seq A
        P_index= [*map(char_map.get, ai)][0] #sending index of P array 
        #print("P_index: ", P_index)
        
        #kernel code
        get_Smat[blockspergrid, threadsperblock](i, d_P[P_index], d_S)
        cuda.synchronize()
        
    end = time.time()
    time_elap.append(end - start)
    print("Time for computing S in GPU with compilation is: {:.2f} secs".format(time_elap[-1]))
    
    #d_S = cuda.to_device(S)
    #cuda.synchronize()
    
    #running cached version
    #start = time.time()
    #for i in range(1, seqA_len):  # 1 to n-1  Length of Seq A
    #    #print("-------------------- Iteration {} --------------------------".format(i))
    #    ai = seqA[i]  #each charof Seq A
    #    P_index= [*map(char_map.get, ai)][0] #sending index of P array 
    #    #print("P_index: ", P_index)
    #    
    #    #kernel code
    #    get_Smat[blockspergrid, threadsperblock](i, d_P[P_index], d_S)
    #    cuda.synchronize()
    #end = time.time()
    #time_elap.append(end - start)
    #print("Time for computing S in GPU after compilation is: {:.2f} secs".format(time_elap[-1]))
    
    print("Total computing time taken for S and P is: {:.2f} secs".format(time_elap[0]+time_elap[-1]))
    
    S = d_S.copy_to_host()
    print("LCS length GPU: ", S[-1][-1])

    if args.valid==True:
        try:
            start = time.time()
            DP= validate_LCS(seqA, seqB, DP)
            end = time.time()
            print("Time for computing S in CPU using DP is: {:.2f} secs".format(end - start))
            
            #print("DP: ", DP)
            assert DP[-1][-1]==S[-1][-1]
            print("\n\n Validation True! LCS length is: {} \n\n".format(S[-1][-1]))
            
        except Exception as e:
            print("\n\n",e.__class__, "LCS don't match! GPU: ", S[-1][-1], " CPU: ",DP[-1][-1])
        

