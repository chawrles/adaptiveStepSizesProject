import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
import time
use_adaptive = False
decreasing = False
use_adaGrad = False
mySeed = 8
np.random.seed(mySeed)

problem = input("Enter 'planes' or 'lasso' problem: ")
while True:
    if problem == "planes" or problem == "lasso":
        break
    problem = input("Invalid problem. Please enter either 'planes' or 'lasso': ")

complexity = input("Enter 'simple' or 'complex' pro blem: ")
while True:
    if complexity == "simple" or complexity == "complex":
        break
    complexity = input("Invalid problem. Please enter either 'simple' or 'complex': ")

if problem == "planes":
    if complexity == "simple":
        set_size = 500
        columns = 2
    elif complexity == "complex":
        set_size = 200
        columns = 10
    A = np.array([np.random.randn(columns)])
    for j in range(set_size - 1):
        A = np.append(A, [np.random.randn(columns)], axis=0)
    b = np.random.randn(set_size, 1)
    x = np.random.randn(columns, 1)

    import math

    def truncate(number, digits) -> float:
        stepper = pow(10.0, digits)
        return math.trunc(stepper * number) / stepper

    def f(x, A, b):
        out_set = np.dot(A, x) + b
        out = np.max(out_set)
        return truncate(out, 8)

    def fPrime(x, A, b, set_size):
        fofx = f(x, A, b)
        grad = (A[i] for i in range(set_size) if truncate(float(np.dot(A[i], x) + b[i]), 8) == fofx)
        return next(grad)

    def find_minimum(A, b):
        x = cp.Variable(columns, 1)
        function = cp.max_entries(A*x + b)
        obj = cp.Minimize(function)
        prob = cp.Problem(obj)
        opt_val = cp.Problem.solve(prob)
        return x.value, opt_val

elif problem == "lasso":
    if complexity == "simple":
        b_size, x_size = 100, 2
    elif complexity == "complex":
        b_size, x_size = 200, 20
    A = np.random.randn(b_size, x_size)
    x = np.random.randn(x_size, 1)
    x[1] = x[1] + 100
    # for element in range(np.size(x)):
    #     if bool(random.getrandbits(1)):
    #         x[element] = 0
    x0 = np.zeros((x_size, 1))
    b = np.dot(A, x)
    noise = np.random.normal(np.mean(b), 0.1, np.shape(b))
    b += noise

    def f(x, A, b, lambd):
        out = 0.5*(np.linalg.norm((np.dot(A,x) - b),2)**2) + lambd*np.linalg.norm(x,1)
        return out

    def fPrime(x, A, b, lambd):
        return np.dot(np.dot(A.T, A),x) - np.dot(A.T, b) + lambd*np.sign(x)

    def find_minimum(A, b):
        x = cp.Variable(x_size,1)
        function = 0.5*(cp.norm(A*x - b)**2) + lambd*cp.norm(x,1) 
        obj = cp.Minimize(function)
        prob = cp.Problem(obj)
        opt_val = cp.Problem.solve(prob)
        return x.value, opt_val

Y, opt = find_minimum(A, b)
    
def gradient_descent_adaptive(A, b, x, num_iterations, alpha, opt, problem, histSize = 10, threshold = 0.5,\
                              use_adaptive = use_adaptive, decreasing = decreasing, use_adaGrad = use_adaGrad):
    originalAlpha, alphaCount, prevAlpha = alpha, 0, alpha
    markAlphaIter, markAlpha, alphaValues = [], [], []
    iterationIndices, objectiveValues = [], []
    avgSearchDirectionSizes, avgSearchIteration = [],[]
    xtraj,ytraj = [],[]
    dfunc_set = []
    reduction = 2
    ag_factor = 1
    for i in range(num_iterations):

        xtraj.append(x[0])
        ytraj.append(x[1])

        if problem == "lasso":
            func = f(x,A,b, lambd)
            dfunc = fPrime(x,A,b, lambd)
            if use_adaGrad:
                x = x - alpha*np.multiply(ag_factor,dfunc)
            else:
                x = x - alpha * dfunc
                
        elif problem == "planes":
            func = f(x,A,b)
            dfunc = fPrime(x,A,b,set_size)
            if use_adaGrad:
                x = x - alpha*np.reshape(np.multiply(ag_factor,dfunc), (columns,1))
            else:
                x = x - alpha * np.reshape(dfunc, (columns,1))

        if i % 1000 == 0:
            print("Difference at " + str(i) + ": " + str(opt-func))
            print("Learning Rate: " + str(alpha))

        iterationIndices.append(i)
        objectiveValues.append(func) 
        alphaValues.append(alpha)
        
        if decreasing:
            alpha = originalAlpha/reduction
            reduction += 1

        dfunc_set.append(dfunc)
        if i >= histSize-1 and np.size(dfunc_set) == histSize * np.size(dfunc):
            if alpha == prevAlpha:
                alphaCount += 1
            else:
                alphaCount = 1
                
            dfunc_unit = [gradient/np.linalg.norm(gradient, ord=2) for gradient in dfunc_set]
            avg_search = np.mean(dfunc_unit, axis = 0)
            avg_search_norm = np.linalg.norm(avg_search, ord=2)
            avgSearchDirectionSizes.append(avg_search_norm)
            avgSearchIteration.append(i)
            
            if use_adaGrad:
                temp = np.sum(np.power(dfunc_set,2), axis = 0)
                ag_factor = 1/np.sqrt(temp)
                
            if use_adaptive:
                prevAlpha = alpha
                if avg_search_norm < threshold or alphaCount == 10000:
                    alpha = originalAlpha/reduction
                    reduction += 1
                    markAlpha.append(func)
                    markAlphaIter.append(i) 
                    dfunc_set = []
                    
            dfunc_set = dfunc_set[1:]
            
            
    print("Finished Training", '\n')
    return x, func, iterationIndices, objectiveValues, avgSearchDirectionSizes, avgSearchIteration, alpha,\
        markAlpha, markAlphaIter, xtraj, ytraj, alphaValues
        
def labelPlots(holdAlpha, animate):
    plt.figure(1)
    plt.title('Avg Search Direction vs Iteration ' + r'$\alpha$ = ' + str(holdAlpha), fontsize=15)
    plt.xlabel('Iterations')
    plt.ylabel('Avg Search Direction')
    if choose == "default":
        plt.legend(['Fixed','Decreasing','AdaGrad','Adaptive'])
    elif choose == "diffHist":
        plt.legend(['histSize=5','histSize=10','histSize=20'])
    
    plt.figure(2)
    plt.title('Value vs Iteration ' + r'$\alpha$ = ' + str(holdAlpha), fontsize=15)
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    if choose == "default":
        plt.legend(['Fixed','Decreasing','AdaGrad','Adaptive'])
    elif choose == "diffHist":
        plt.legend(['histSize=5','histSize=10','histSize=20'])
    
    plt.figure(3)
    plt.title('Relative Error vs Iteration ' + r'$\alpha$ = ' + str(holdAlpha), fontsize=15)
    plt.xlabel('Iterations')
    plt.ylabel('Relative Error')
    if choose == "default":
        plt.legend(['Fixed','Decreasing','AdaGrad','Adaptive'])
    elif choose == "diffHist":
        plt.legend(['histSize=5','histSize=10','histSize=20'])
    if not animate and complexity == "simple" and choose == "custom":
        plt.figure(4)
        plt.title('xtraj vs y traj ' + r'$\alpha$ = ' + str(holdAlpha), fontsize=15)
        plt.xlabel('x')
        plt.ylabel('y')
    
    plt.figure(5)
    plt.title('Alpha Value vs Iteration', fontsize=15)
    plt.xlabel('Iterations')
    plt.ylabel('Alpha')
    plt.title('Alpha vs Iteration ' + r'$\alpha$ = ' + str(holdAlpha), fontsize=15)
    if choose == "default":
        plt.legend(['Fixed','Decreasing','AdaGrad','Adaptive'])
    elif choose == "diffHist":
        plt.legend(['histSize=5','histSize=10','histSize=20'])

def trainPrintPlot(problem, A, b, x, Y, opt, histSize=10, numPrintPlots=2, alpha=1, threshold=0.1,\
                   num_iterations=10000, compare=True, use_adaptive=False, decreasing=False, use_adaGrad=False,\
                   saveName="", animate=False):
    plt.close('all')
    holdAlpha = alpha
    decreasing = decreasing
    for amount in range(numPrintPlots):
        if compare and choose != "diffHist":
            if amount == 0:
                use_adaptive = False
            elif amount == 1:
                decreasing = True
                use_adaptive = False
            elif amount == 2:
                decreasing = False
                use_adaGrad = True
            elif amount == 3:
                use_adaptive = True
                use_adaGrad = False
            elif amount == 4:
                use_adaGrad = True
        elif compare and choose == "diffHist":
            if amount == 0:
                use_adaptive = True
                histSize = 5
            elif amount == 1:
                histSize = 10
            elif amount == 2:
                histSize = 15
                
        result, funct, iterationIndices, objectiveValues, avgSearchDirectionSizes, avgSearchIteration, alpha, markAlpha,\
            markAlphaIter, xtraj, ytraj, alphaValues\
            = gradient_descent_adaptive(A,b,x,num_iterations, alpha = holdAlpha, opt = opt, problem = problem,histSize = histSize,\
                                        use_adaptive = use_adaptive, use_adaGrad = use_adaGrad, threshold = threshold,\
                                        decreasing = decreasing)

        print("Predicted Final Value: " + str(funct))
        print("True Final Value: " + str(opt), '\n')
        print("Predicted Optimal Point: " + str(result))
        print("True Final Optimal Point: " + str(Y), '\n')
        
        relativeError = abs(np.subtract(objectiveValues,opt))/abs(opt)
        
        plt.figure(1)
        avgSearchLines, = plt.plot(avgSearchIteration, avgSearchDirectionSizes, linestyle='-', label = str(amount), alpha = 0.7)
        plt.figure(2)
        valueLines, = plt.plot(iterationIndices, objectiveValues,linestyle='-', label = str(amount))
        #plt.plot(markAlphaIter, markAlpha, color='r', marker='x', linestyle="None")
        plt.figure(3)
        relErrorLines, = plt.semilogy(iterationIndices, relativeError,linestyle='-', label = str(amount), alpha = 0.8)
        #plt.plot(markAlphaIter, markAlpha, color='r', marker='x', linestyle="None")
        
        if animate:
            plt.figure(4)
            xtraj = xtraj[:800]
            ytraj = ytraj[:800]
            fig, ax = plt.subplots()
            if problem == "planes":
                plt.ylim(-0.19, 0.063)
                plt.xlim(0.28,1.09)
            elif problem == "lasso":
                plt.ylim(99.699, 99.94)
                plt.xlim(-0.075,0.031)
            line, = ax.plot(xtraj, ytraj, color='black', linestyle='-', marker='o', markerfacecolor='red',\
                            markersize=5, alpha=0.5)
            def update(num, xtraj, ytraj, line):
                line.set_data(xtraj[:num], ytraj[:num])
                return line,
            ani = animation.FuncAnimation(fig, update, len(xtraj), fargs=[xtraj, ytraj, line],\
                                  interval=50, blit=True, repeat = True)
            ani.save(saveName + '.gif')
        elif complexity == "simple" and choose == "custom":
            plt.figure(4)
            if problem == "planes":
                plt.ylim(-0.19, 0.063)
                plt.xlim(0.28,1.09)
            elif problem == "lasso":
                plt.ylim(99.699, 99.94)
                plt.xlim(-0.075,0.031)
            plt.plot(xtraj, ytraj, color='black', linestyle='-', marker='o', markerfacecolor='red', markersize=5, alpha=0.5)
            
        plt.figure(5)
        alphaValues, = plt.semilogy(iterationIndices, alphaValues,linestyle='-', label = str(amount))

    labelPlots(holdAlpha, animate)
    plt.show()

choose = input("'default', 'diffHist', or 'custom' input? ")
while True:
    if choose == "default" or choose == 'diffHist' or choose == "custom":
        break
    choose = input("Invalid input. Please enter 'default', 'diffHist', or 'custom': ")

if choose == "default":
    if problem == "planes":
        alpha = 0.5
    elif problem == "lasso":
        alpha = 0.0005
    trainPrintPlot(problem,A,b,x,Y,opt,numPrintPlots=4, alpha=alpha, num_iterations=10000, threshold = 0.5, compare= True)
    
elif choose == "diffHist":
    if problem == "planes":
        alpha = 0.5
    elif problem == "lasso":
        alpha = 0.0005
    trainPrintPlot(problem,A,b,x,Y,opt,numPrintPlots=3, alpha=alpha, num_iterations=10000, threshold = 0.5, compare= True)
    
elif choose == "custom":
    numPrintPlots = 1
    while True:
        try: 
            alpha = float(input("Alpha Value: "))
        except ValueError: 
            print("Not a valid number.")
            continue
        else: break
    while True:
        try: 
            num_iterations = int(input("# of iterations: "))
        except ValueError: 
            print("Not a valid number.")
            continue
        else: break
    while True:
        try: 
            threshold = float(input("Threshold: "))
        except ValueError: 
            print("Not a valid number.")
            continue
        else: break
    chooseAlg = input("fixed, decreasing, adaGrad, adaptive, or hybrid? ")
    while True:
        if chooseAlg == "fixed" or chooseAlg == "decreasing" or chooseAlg == "adaGrad" or chooseAlg == "adaptive"\
            or chooseAlg == "hybrid":
            break
        chooseAlg = input("Invalid input. Please enter 'fixed', 'decreasing', 'adaGrad', 'adaptive', or 'hybrid': ")
    if chooseAlg == "decreasing":
        decreasing = True
    elif chooseAlg == "adaGrad":
        use_adaGrad = True
    elif chooseAlg == "adaptive":
        use_adaptive = True
    elif chooseAlg == "hybrid":
        use_adaptive = True
        use_adaGrad = True

    animate = input("Animate? ")
    while True:
        if animate == "yes" or animate == "no":
            break
        animate = input("Invalid input. Please enter 'yes' or 'no': ")
                        
    if animate == "yes":
        animate = True
        saveName = input("Name animation save: ")
    elif animate == "no":
        animate = False
        saveName = ""
                        
    trainPrintPlot(problem,A,b,x,Y,opt,numPrintPlots=numPrintPlots,alpha=alpha,num_iterations=num_iterations,threshold=threshold,\
                   compare=False, decreasing=decreasing, use_adaGrad=use_adaGrad,use_adaptive=use_adaptive, animate=animate,\
                  saveName=saveName)
       