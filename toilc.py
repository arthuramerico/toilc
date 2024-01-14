import cvxpy as cp
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys




"""This code implements the LP problem discussed in the document 
'A Brief Intro to Quantitative Information Flow and
its Relation to our Problem'
"""


#######################################################################
"""First, we implement functions to generate 
the desired  variables, inputs and channels"""

def probOfX(n,stdev,samples, distX, delta,deltaLeft):
    """This fuctions returns a normalized probability distribution with 
    input size n, drawn from sampling 10**samples times a normal distribution
    with median n-1/2 and standard deviation stdev"""
    
    #First, we check if the user input a probability distribution, and use it in  that case 
    if distX is not None:
        if isinstance(distX, np.ndarray):
            px=distX
        elif isinstance(distX,list):
            px=np.array(distX)
        elif isinstance(distX, str):
            f=open(distX, 'r')
            px=np.loadtxt(f)
            f.close()
        else:
            px=np.loadtxt(distX)
            distX.close()
        if px.ndim>1:
                raise Exception("Input distribution must be a single row or column")
        px=px/px.sum() #Normalizing so px is a probability distribution
        n=px.shape[0]
    else:
        #Otherwise, we generate our own probability distribution
        normalSampling=np.random.normal((n-1)/2,stdev,size=int(10**samples))
        normalRounding=(np.rint(normalSampling)).astype(int)    
        value, counts = np.unique(normalRounding, return_counts=True)
        sampledFrequency=dict(zip(value, counts))
        unormalizedPx= np.array([sampledFrequency[i] if i in sampledFrequency else 0 for i in range(n)])
        px=unormalizedPx/unormalizedPx.sum()

    #Finally, we calculate the value of cutoff:
    cutoff=n
    cumsum=0
    while  cutoff>0 and cumsum+px[cutoff-1]<delta : #This guarantees that the sum of cutoff and above are AT MOST delta, instead of at least
        cutoff-=1
        cumsum+=px[cutoff]
    cutoffLow=0
    cumsum=0
    while  cutoffLow<n-1 and cumsum+px[cutoffLow]<deltaLeft : #This guarantees that the sum up to  cutoffLow is AT MOST deltaLeft, instead of at least
        cumsum+=px[cutoffLow]
        cutoffLow+=1
    return px, n, cutoff,cutoffLow

def redundancyChannel(n):
    """This implements the redudancy channel X -> X^2.  ]
    The outputs are ordered in lexicographical ordering"""
    redundancy=np.zeros((n,n*n))
    for i in range(n):
        redundancy[i,n*i+i]=1
    return redundancy


def publicChannelAddition(n,m):
    """This implements the public channel in the additive scenario, with input 
    size n and m alice action size"""
    #If the user input a public channel, we use it
    #Otherwise, we generate one
    public=np.zeros((n*m,n))
    for i in range(n):
        for j in range(m):
            public[m*i+j, min(i+j,n-1)]=1
    return public    

def publicChannel(n,m, publicMatrix):
    """This implements the public channel in the additive scenario, with input 
    size n and m alice action size"""
    #If the user input a public channel, we use it
    if publicMatrix is not None:
        if isinstance(publicMatrix, np.ndarray):
            public=publicMatrix
        elif isinstance(publicMatrix,list):
            public=np.array(publicMatrix)
        elif isinstance(publicMatrix, str):
            f=open(publicMatrix, 'r')
            public=np.loadtxt(f)
            f.close()
        else:
            public=np.loadtxt(publicMatrix)
            publicMatrix.close()
        if public.shape[0] !=n*m: #Checks if the user made a mistake with the channel sizes
            raise Exception("Public channel must have input size equal to the range of X times action set")
    else:
        #Otherwise, we generate one
        public=publicChannelAddition(n,m)
    return public    

def identityChannel(n):
    """This implements a identity channel"""
    return np.identity(n)

def auxChannel(n,q,auxMatrix,printAux,printAuxFile,pretty):
    """This function returns the Aux channel, with n being the input size and 
    q the noise given by the user, ranging from 0 (no noise) to 1"""
    #if the user inputs 0 (or any lower number) as noise, we return the transparent channel

    #If the user inputs an aux channel, we use it
    if auxMatrix is not None:
        if isinstance(auxMatrix, np.ndarray):
            aux=auxMatrix
        elif isinstance(auxMatrix,list):
            aux=np.array(auxMatrix)
        elif isinstance(auxMatrix, str):
            f=open(auxMatrix, 'r')
            aux=np.loadtxt(f)
            f.close()
        else:
            aux=np.loadtxt(auxMatrix)
            auxMatrix.close()
        if aux.shape[0] !=n: #Checks if the user made a mistake with the channel sizes
            raise Exception("Aux channel must have input size equal to inputSize")
    else:    
        #If the user did not input a channel, we create one with the value of noise
        if q<0 or np.isclose(q,0):
            aux=np.identity(n)
        #if the user inputs 1 (or any greater number) as noise, we return the null channel
        if q>1 or np.isclose(q,1):
            aux=np.array([[1]*n]).T

        #If noise is between one and 0, we return a noisy channel based on the geometric distribution
        else:
            auxCh=np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    if i==j:
                        auxCh[i,j]=1-q
                    else:
                        auxCh[i,j]=(1-q)*((q)**abs(i-j))
            #This normalizes the channel
            aux=(np.divide(auxCh.T,auxCh.sum(axis=1))).T
    #If the user asks for it, we print aux     
    if printAux:
        print("Aux Channel:\n")
        print(aux)
    if printAuxFile is not None:
        if pretty:
            np.savetxt(printAuxFile,aux, fmt="%0.3f")
        else:
            np.savetxt(printAuxFile,aux)
        printAuxFile.close()
    return aux


def checkForErrors(inputSize,stdev,sizeAlice,eEpsilon,delta, deltaLeft, tailMult):
    """This checks for errors in the variables"""
    if inputSize<=0:
        raise Exception("The value of 'inputSize' must be positive")
    if stdev<=0:
        raise Exception("The value of 'stdev' must be positive")
    if sizeAlice<=0:
        raise Exception("The value of 'act' must be positive")
    if eEpsilon<1:
        raise Exception("The value of e^epsilon must be at least 1")
    if delta>1 or delta<0:
        raise Exception("Delta must be a value between 0 and 1")
    if deltaLeft>1 or deltaLeft<0:
        raise Exception("DeltaLeft must be a value between 0 and 1")
    if tailMult<1:
        raise Exception("tailMult must be at least 1")



"""Finally, a printing function and a plotting function:"""

def printFinal(inputSize, px, pTildex, expectedAlice, maximumAlice, varreal, alice,vecProbAlice,
               threshold, outputX, omitSolution, printAlice, printAliceFile, pretty, cutoff, cutoffLow):

    #This will be used to write the output file and for plotting
    if outputX is not None:
        t = [i for i in range(inputSize)]
        outputX.write("X px"+ " "*20+" X~\n")
        for i in t:
            s=f"{i} {px[i]} {pTildex[i]}\n"
            outputX.write(s)
        outputX.close

    rowOfIndexes = np.array([i for i in range(inputSize)]) 
    if not omitSolution:
        print("Expected Value of X:", np.dot(rowOfIndexes,px))
        print("Expected Value of X~:", np.dot(rowOfIndexes,pTildex))
        print("Expected Value of Alice's actions:", expectedAlice)
        if not np.isclose(threshold,1):
            print("Optimal expected value of Alice's actions", maximumAlice)
        print("Variance of Alice's actions:", varreal)
        print("Cutoff point for the left tail:", cutoffLow)
        print("Cutoff point for the right tail:", cutoff)
            
    if printAlice:
        print("Alice's strategy:\n", alice)
        print("Alice's marginal probability distribution: \n")
        print(vecProbAlice)
    if printAliceFile is not None:
        if pretty:
            np.savetxt(printAliceFile,alice, fmt="%0.3f")
        else:
            np.savetxt(printAliceFile,alice)
        printAliceFile.close()


def plotResults(inputSize,px,eEpsilon,pTildex, noPlot, filename):
    eMinusEpsilon=1/eEpsilon
    t = [i for i in range(inputSize)]
    plt.plot(t, px, '--', label=r'$p_X$')
    plt.plot(t,eEpsilon*px,'--',label=r'$e^\epsilon p_{X}$')
    plt.plot(t,eMinusEpsilon*px,'--',label=r'$e^{-\epsilon} p_{X}$')
    plt.plot(t, pTildex,'--', label=r'$p_{\tilde{X}}$')
    plt.xlabel("Volume Pressure")
    plt.ylabel("Probability Mass")
    plt.legend(loc="upper left")
    if len(filename)>0:
        if len(filename)>4 and filename[-4:]=='.pdf':
            plt.savefig(filename)
        else:
            plt.savefig(filename+'.pdf')
    if not noPlot:
        plt.show()
    plt.clf()



"""We also implement a class Solution, which will be returned to the user"""

class Solution(object):
    """docstring for Solution"""
    def __init__(self, expectedX, expectedXtilde, expectedAlice, maximumAlice, varianceAlice, cutoff,cutoffLow):
        self.expectedX = expectedX
        self.expectedXtilde = expectedXtilde
        self.expectedAlice = expectedAlice
        self.maximumAlice= maximumAlice
        self.varianceAlice=varianceAlice
        self.cutoff=cutoff
        self.cutoffLow=cutoffLow




###########################################################################
"""We define a main function that can be called if one uses this as a module"""

def solve(inputSize=51, std=5, samples=7., act=20, noise=0., eeps=1.3, threshold=1.0, delta=0.01, deltaLeft=0, tailMult=1.1, #Problem Variables
         maxExpected=False, noVariance=False,   #Problem behaviour options
         printAux=False, printAuxFile=None, printAlice= False, printAliceFile= None,
         pretty=False, omitSolution=False, outputX= None, noPlot=False, pdfOutput="", #Printing options
         distX=None, auxMatrix=None, publicMatrix=None): #User input channels
    
    #First, some small things to set before we get to the meat of the algorithm
    checkForErrors(inputSize, std, act, eeps, delta, deltaLeft, tailMult) #This raises exceptions if the inputs are not compatible
    

    #This prints the whole channels/distributions on the screen
    np.set_printoptions(threshold=sys.maxsize)
    #If the user wants pretty options, we do as he wishes
    if pretty:
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)}) # This gives more legible printing


    ##################################################################################################################
    """This part of the script initialises the channels and probability distributions"""
    px, inputSize, cutoff,cutoffLow=probOfX(inputSize,std,samples,distX,delta,deltaLeft) #cutoff is the value from which the right-hand tail is ignored
    aux=auxChannel(inputSize, noise, auxMatrix, printAux, printAuxFile, pretty)
    redundancy=redundancyChannel(inputSize)
    I=identityChannel(inputSize)
    public=publicChannel(inputSize, act, publicMatrix)

    #we also, for ease of notation, define e^-epsilon
    eMinusEpsilon=1/eeps



    ########################################################################################################################
    """This implements the first linear program, which establishes the optimum value of E[X~]"""

    # Creates Alice's channel as the variable of the LP
    alice = cp.Variable((aux.shape[1],act))


    """Using CVXPY operators (available at https://www.cvxpy.org/tutorial/functions/index.html),
    our system (as depicted in the documents is simply"""

    system=redundancy @(cp.kron(I,aux@alice))@public

    #the vector p_\~X is thus
    rowOnes=np.ones(inputSize)#this is just an auxiliary vector 
    pTildex= rowOnes @ cp.diag(px) @ system 


    # Create constraints of the LP 
    constraints = [alice>=0,
                        alice<=1,
                        cp.sum(alice,axis=1)==1,        #These first three make alice a channel
                        pTildex[cutoffLow:cutoff] <=eeps*px[cutoffLow:cutoff] + 1e-6,          # upper bound of diff. privacy. The 1e-6 is a tolerance parameter for the solver
                        pTildex[cutoffLow:cutoff] >=eMinusEpsilon*px[cutoffLow:cutoff] - 1e-6,      # lower bound of diff. privacy
                        cp.sum(pTildex[cutoff:])<=tailMult*delta +1e-6  #Bound on the cumulative distribution of the tail
                        ]
    

    #These vectors are necessary for calculating expected values
    rowOfIndexes = np.array([i for i in range(inputSize)]) 
    rowOfIndexes.reshape(1,inputSize) 

    rowOfIndexesAlice = np.array([i for i in range(act)]) 
    rowOfIndexesAlice.reshape(1,act)

    """We now create variables to calculate Alice's variance and expected value"""

    #A few more necessary vectors and channels:
    rowAlice=np.ones(act)
    identityAlice=np.eye(act)

    #vector of Alice's probabilities:
    vecProbAlice=rowOnes @ cp.diag(px) @ aux @ alice

    #expected value of Alice:
    expectedAlice=vecProbAlice @ (rowOfIndexesAlice.T)

    #Actual Variance of Alice:
    varreal=vecProbAlice@cp.square(rowOfIndexesAlice - expectedAlice)



    # Now we implement the objective function  

    if not maxExpected: 
        #Unless the user asks us to, we maximize Alice's actions
        obj = cp.Maximize(rowOnes @ cp.diag(px) @ aux @ alice @ (rowOfIndexesAlice.T) ) 
    else:
        #Maximizes the expected value of \~X 
         obj = cp.Maximize(pTildex @ (rowOfIndexes.T)) 


    # Finally, we solve the LP
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCIPY)
    if prob.status !="optimal":
        raise Exception("The LP could not be solved")


    """If the user does not want to run the minimizing Variance step, we stop the script here"""
    if noVariance:

        printFinal(
        inputSize, 
        px, 
        pTildex.value,
        expectedAlice.value,
        expectedAlice.value,
        varreal.value,
        alice.value,
        vecProbAlice.value,
        threshold,
        outputX,
        omitSolution,
        printAlice,
        printAliceFile,
        pretty,
        cutoff,
        cutoffLow
        )

        #Now we plot the results

        #This creates the appropriate name for the file
        if not noPlot or len(pdfOutput)>0:
            plotResults(inputSize,px,eeps,pTildex.value, noPlot, pdfOutput)

        #And we exit the script:
        return Solution(np.dot(rowOfIndexes,px), np.dot(rowOfIndexes,pTildex.value), expectedAlice.value, expectedAlice.value, varreal.value, cutoff, cutoffLow)





    """Finally, we store some values for the next LP, which will minimize Alice's actions' variance"""

    #The maximum expected value of Alice's actions
    maximumAlice=expectedAlice.value

    #######################################################################################################

    """Now we start the second optimization, which will minimize variance according to the -threshold parameter"""

    #First, we need to repeat some of the the matrices created for the first LP"""

    alice2 = cp.Variable((aux.shape[1],act))
    system2=redundancy @(cp.kron(I,aux@alice2))@public
    pTildex2= rowOnes @ cp.diag(px) @ system2 

    #vector of Alice's probabilities:
    vecProbAlice2=rowOnes @ cp.diag(px) @ aux @ alice2

    #expected value of Alice:
    expectedAlice2=vecProbAlice2 @ (rowOfIndexesAlice.T)

    #PseudoVariance of Alice:
    var=vecProbAlice2@np.square(rowOfIndexesAlice - maximumAlice)

    #Actual Variance of Alice:
    varreal2=vecProbAlice2@cp.square(rowOfIndexesAlice - expectedAlice2)




    # Create constraints
    constraints2 = [alice2>=0,
                        alice2<=1,
                        cp.sum(alice2,axis=1)==1,        #These first three make alice a channel
                        pTildex2[cutoffLow:cutoff] <=eeps*px[cutoffLow:cutoff] + 1e-6,          # upper bound of diff. privacy
                        pTildex2[cutoffLow:cutoff] >=eMinusEpsilon*px[cutoffLow:cutoff] - 1e-6,      # lower bound of diff. privacy
                        #This last one will give the constraint as a percentage of Alice's startegy's maximum value
                        expectedAlice2>=threshold*maximumAlice - 1e-5, #This is a solver tolerance parameter
                        cp.sum(pTildex2[cutoff:])<=tailMult*delta +1e-6  #Bound on the cumulative distribution of the tail
                        ]

   

    #This makes the objective to minimize the pseudo-variance
    obj2= cp.Minimize(var)


    # Now we implement and solve the problem.
    prob2 = cp.Problem(obj2, constraints2)
    prob2.solve(solver=cp.SCIPY)  
    if prob2.status !="optimal":
        raise Exception("The LP could not be solved")
    #This prints on screen and files what the user asked to
    printFinal(
    inputSize, 
    px, 
    pTildex2.value,
    expectedAlice2.value,
    maximumAlice,
    varreal2.value,
    alice2.value,
    vecProbAlice2.value,
    threshold,
    outputX,
    omitSolution,
    printAlice,
    printAliceFile,
    pretty,
    cutoff,
    cutoffLow
    )

    #Now we plot the results

    #This creates the appropriate name for the file
    if not noPlot or len(pdfOutput)>0:
        plotResults(inputSize,px,eeps,pTildex2.value, noPlot, pdfOutput)

    return Solution(np.dot(rowOfIndexes,px), np.dot(rowOfIndexes,pTildex2.value), expectedAlice2.value, maximumAlice, varreal2.value, cutoff, cutoffLow)

if __name__ == "__main__":
   
    #######################################################################
    """If this is being used as a script, This part of the code implements the optional arguments the user may give
    We use argparse to receive user inputs. If those are not given, we use some standart value
    The user can obtain help regarding these options by typing the argument -h on the command line
    """
    parser = argparse.ArgumentParser(description="Solves the LP for Alice's strategy")


    #The problem variables
    parser.add_argument("-inputSize",  dest="inputSize", default = 51, help="Range of X axis", type= int)
    parser.add_argument("-std", dest="std", default = 5, help="Standard deviation of X", type = int)
    parser.add_argument("-samples", dest ="samples", default = 7, help="log10 of the number of samples", type=float)
    parser.add_argument("-act", dest ="actions", default = 20, help="number of Alice's actions", type=int)
    parser.add_argument("-noise",dest ="noise", default = 0., help="How noisy is the aux channel, from 0 to 1", type=float)
    parser.add_argument("-eeps", dest ="eeps", default = 1.3, help="How much is e^epsilon", type=float)
    parser.add_argument("-maxExpected", action='store_true', help="maximizes E[X~] instead of Alice's actions")
    parser.add_argument("-noVariance", action='store_true', help="Suppress the minimizing variance procedure")
    parser.add_argument("-threshold", dest ="threshold", default=1.0, help="Redefines the minimum threshold for the expected value of Alice's as the given ratio of the optimal", type= float)
    parser.add_argument("-delta",  dest="delta", default = 0.01, help="Value of delta for the right tail", type= float)
    parser.add_argument("-deltaLeft",  dest="deltaLeft", default = 0, help="Value of delta for the left tail", type= float)
    parser.add_argument("-tailMult",  dest="tailMult", default = 1.1, help="Value of the allowed multiplier for the cumulative distribution after the tail", type= float)


    #Optional printing:

    parser.add_argument('-printAux', action='store_true', help="Prints the aux channel in the prompt") #prints aux channel
    parser.add_argument('-printAlice', action='store_true', help="Prints the alice channel in the prompt") #prints Alice's strategy
    parser.add_argument('-printAuxFile', dest ="printAuxFile", type=argparse.FileType('w', encoding='UTF-8'), help="Prints the aux channel in the file given" ) #prints aux channel in a file
    parser.add_argument('-printAliceFile', dest ="printAliceFile", type=argparse.FileType('w', encoding='UTF-8'), help="Prints the aux channel in file given") #prints Alice's strategy in a file
    parser.add_argument('-pretty', action='store_true', help="truncates all outputs to 3 decimal plates") #Formats everything to "%0.3f"
    parser.add_argument('-omitSolution', action='store_true', help="Omits the expected values of p_X, p_X~ and Alice's actions")
    parser.add_argument("-outputX",dest ="outputX", type=argparse.FileType('w', encoding='UTF-8'), help="Prints p_X and p_X~ in a indicated output file")



    #Omit plotting or saving pdf:

    parser.add_argument('-noPlot', action='store_true', help="Omits the plot") #omits plotting
    parser.add_argument("-pdfOutput", dest ="pdfOutput", default="", help="Name of the file to save the plot as pdf")

    #Optional input, aux and public channels
    parser.add_argument("-distX", dest ="distX", type=argparse.FileType('r'), help="Loads p_X from a given input file")
    parser.add_argument("-auxMatrix", dest ="auxMatrix", type=argparse.FileType('r'), help="Loads aux channel from a given input file")
    parser.add_argument("-publicMatrix", dest ="publicMatrix", type=argparse.FileType('r'), help="Loads public channel from a given input file")

    args=parser.parse_args()


    solve(inputSize=args.inputSize, std=args.std, samples=args.samples, act=args.actions, noise=args.noise,eeps=args.eeps, 
         threshold=args.threshold, delta=args.delta, deltaLeft=args.deltaLeft, maxExpected=args.maxExpected, noVariance=args.noVariance, 
         tailMult=args.tailMult, printAux=args.printAux, printAuxFile=args.printAuxFile, printAlice=args.printAlice, printAliceFile=args.printAliceFile, 
         pretty=args.pretty, omitSolution=args.omitSolution, outputX= args.outputX, noPlot=args.noPlot, 
         pdfOutput=args.pdfOutput, distX=args.distX, auxMatrix=args.auxMatrix, publicMatrix=args.publicMatrix)
