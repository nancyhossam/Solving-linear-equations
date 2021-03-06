from tkinter import *
from tkinter import filedialog, StringVar
import re
from timeit import default_timer as timer
import numpy as np
import pprint
import matplotlib.pyplot as plt

root = Tk()
root.title("Solving Linear Systems")
root.geometry("1000x500")

#matrix = [[3, 2, 4, 1, 8], [0, 1, -1, 6, 8], [0, 0, 6, 1, 1], [0, 0, 0, 3, 3]]
matrix = []
numberofrows = 0

percision = StringVar()
maxitr = StringVar()
N = StringVar()
op = StringVar()

method = StringVar()
InputType = StringVar()
isall = StringVar()

m = ''
vars = []
inital_points = []
inital = []


def init_input():
    str5 = 'Data Entered By User'
    if InputType.get() == str5:
        global numberofrows
        numberofrows = int(NoEqEntry.get())
        if (numberofrows > 0):
            global inital
            inital= []
            inits = []
            for idx in range(int(numberofrows)):
                i0 = StringVar()
                inits.append(Entry(root, textvar=i0))
                inits[idx].grid(row=4 + int(idx), column=1)
                inital.append(i0)

def user_input():
    str5 = 'Data Entered By User'
    if InputType.get() == str5:
        global numberofrows
        numberofrows = int(NoEqEntry.get())
        print(numberofrows)
        global matrix
        if (numberofrows > 0):
            global listofentries
            global vars
            vars = []
            listofentries = []
            for idx in range(int(numberofrows)):
                var = StringVar()
                listofentries.append(Entry(root, textvar=var))
                listofentries[idx].grid(row=4 + int(idx), column=0)
                vars.append(var)


def temp1():
    str5 = 'Data Entered By User'
    if InputType.get() == str5 and method.get()=='Gauss-Seidal':
        v = ["" for x in range(int(numberofrows))]
        global inital_points
        initial_points = ["" for x in range(int(numberofrows))]
        for i in range(numberofrows):
            print(vars[i].get())
            v[i] = vars[i].get()
            inital_points.append(inital[i].get())
        Make_matrix(v)
        #for n in range(numberofrows):
           # initial_points[n] = float(inital_points[n])
def temp2():
    str5 = 'Data Entered By User'
    if InputType.get() == str5:
        v = ["" for x in range(int(numberofrows))]
        for i in range(numberofrows):
            print(vars[i].get())
            v[i] = vars[i].get()
        Make_matrix(v)

def get_variables(equations):
    all_var = set()
    for eq in equations:
        var = set(re.findall('[a-zA-z]', eq))
        all_var = all_var.union(var)
    return sorted(list(all_var)) + ['constant']


def parse(equation):
    parsed = re.sub('[*]', '', equation)
    parsed = re.sub(' [+] ', ' ', parsed)
    parsed = re.sub(' [-] ', ' -', parsed)
    parsed = re.sub('[-] ', '-', parsed)
    parsed = parsed.split(' ')
    print(parsed)
    var2coeff = {}

    for term in parsed:

        if term[-1].isalpha():
            check = ""
        else:
            check = "-"
        if term[-1].isalpha():
            if term[:-1] == check:

                coeff = 1
            else:
                if term[:-1] == "-":
                    coeff = -1
                else:
                    coeff = float(term[:-1])
            var2coeff[term[-1]] = coeff
        else:
            var2coeff['constant'] = float(term)
    return var2coeff


def Make_matrix(equations):
    all_var = get_variables(equations)
    global matrix
    matrix = np.zeros((len(equations), len(all_var)))
    var2idx = {}
    for i, var in enumerate(all_var):
        var2idx[var] = i

    for i, eq in enumerate(equations):
        var2coeff = parse(eq)
        for var, coeff in var2coeff.items():
            j = var2idx[var]
            matrix[i, j] = coeff
    print(matrix)


def fileinput():
    str4 = 'Data Entered From File'
    if InputType.get() == str4:
        print('in file')
        f = open("test.txt", "r")
        global numberofrows
        n = f.readline()
        numberofrows = int(n)
        global m
        m = f.readline()
        equations = ["" for x in range(int(n))]
        initial_points = ["" for x in range(int(n))]
        count = 0
        while True:
            if count == int(n):
                break;
            line = f.readline()
            if not line:
                break
            equations[count] = line
            count += 1
        line = f.readline()
        if not line:
            global inital_points
            initial_points = ["" for x in range(int(n))]
        else:
            global inital_points
            initial_points = line
            print(line)
        Make_matrix(equations)
        print('initial file')
        print(initial_points)
        print(m)
        return matrix

def write_output(x):
    f = open("demofile3.txt", "w")
    f.write(" The roots are:\n")
    for i in range(len(x)):
        f.write(str(x[i]))
        f.write(" ")
    f.close()


def Gauess_Elimination(a, n):
    str1 = 'Gaussian-elimination'
    if method.get() == str1 or isall.get() == 'Run All Methods':
        start_time = timer()
        xe = np.zeros(n)
        for i in range(n):
            if a[i][i] == 0.0:
                sys.exit('Divide by zero detected!'
                         '')

            for j in range(i + 1, n):
                ratio = a[j][i] / a[i][i]

                for k in range(n + 1):
                    a[j][k] = a[j][k] - ratio * a[i][k]

        # Back Substitution
        xe[n - 1] = a[n - 1][n] / a[n - 1][n - 1]

        for i in range(n - 2, -1, -1):
            xe[i] = a[i][n]

            for j in range(i + 1, n):
                xe[i] = xe[i] - a[i][j] * xe[j]

            xe[i] = xe[i] / a[i][i]
        write_output(xe)
        end_time = timer()
        time_taken = (end_time - start_time) * 1000
        # Displaying solution
        print('\nRequired solution is: ')
        for i in range(n):
            print('X%d = %0.2f' % (i, xe[i]), end='\t')
        output = Label(root, text="Gausse Elimination Solution: ", relief="solid", font=("ariel", 9, "bold"))
        output.grid(row=10, column=0)
        w = Text(root, width=20, height=2)
        w.grid(row=10, column=1)
        w.insert(END, xe)
        exec = Label(root, text="Execution Time taken " + str(time_taken) + " msec", relief="solid",
                     font=("ariel", 9, "bold"))
        exec.grid(row=11, column=1)


def Gauess_jordan(a, n):
    str2 = 'Gaussian-jordan'
    if method.get() == str2 or isall.get() == 'Run All Methods':
        start_time = timer()
        Ea = 0.001
        errdigits = 0
        while Ea < 1:
            Ea = Ea * 10
            errdigits += 1
        x = np.zeros(n)

        # Applying Gauss Jordan Elimination
        for i in range(n):
            if a[i][i] == 0.0:
                sys.exit('Cannot use Gausses Jordan method Divide by zero detected!')
            if a[i][i] != 1:
                y = a[i][i]
                for z in range(n + 1):
                    a[i][z] = a[i][z] / y

            for j in range(n):
                if i != j:
                    rat = a[j][i]
                    for k in range(n + 1):
                        a[j][k] = a[j][k] - rat * a[i][k]

        # Obtaining Solution
        for k in range(n):
            if a[k][k] == 0:
                if a[k][n] == 0:
                    sys.exit('Infinite number of solutions')
                else:
                    sys.exit('No solution')
        print(a)
        for i in range(n):
            print("hi")
            x[i] = a[i][n]
            print(x[i])
        write_output(x)
        end_time = timer()
        time_taken = (end_time - start_time) * 1000
        rounded = np.round(x, decimals=errdigits)
        # Displaying solution
        print('\nRequired solution is: ')
        print("\nRounded values : \n", rounded)
        output = Label(root, text="Gausse Jordan Solution: ", relief="solid", font=("ariel", 9, "bold"))
        output.grid(row=12, column=0)
        exec = Label(root, text="Execution Time taken " + str(time_taken) + " msec", relief="solid",
                     font=("ariel", 9, "bold"))
        exec.grid(row=13, column=1)
        w = Text(root, width=20, height=2)
        w.grid(row=12, column=1)
        w.insert(END, rounded)


def decompose(mat):
    str3 = 'LU decomposition'
    if method.get() == str3 or isall.get() == 'Run All Methods':
        if mat[0][0] == 0:
            print("Division by zero!")
        else:
            start_time = timer()
            n = len(mat)
            print(n)
            print(len(mat[0]))
            l = len(mat[0]) - 1
            A = [[0 for i in range(0, l)] for j in range(0, n)]
            b = [0 for i in range(n)]

            # Separate A values
            for i in range(0, n):
                for j in range(0, l):
                    A[i][j] = mat[i][j]

            # Separate b values
            for i in range(0, n):
                b[i] = mat[i][n]

            # Fill L matrix and its diagonal with 1
            L = [[0 for i in range(n)] for i in range(n)]
            for i in range(0, n):
                L[i][i] = 1

            # Fill U matrix
            U = [[0 for i in range(0, n)] for i in range(n)]
            for i in range(0, n):
                for j in range(0, n):
                    U[i][j] = A[i][j]

            for i in range(0, n):
                for k in range(i + 1, n):
                    c = -U[k][i] / float(U[i][i])
                    L[k][i] = -c  # Store the multiplier
                    for j in range(i, n):
                        U[k][j] += c * U[i][j]  # Multiply with the pivot line and subtract
                for k in range(i + 1, n):
                    U[k][i] = 0

            # Perform substitution Ly=b
            y = [0 for i in range(n)]
            for i in range(0, n, 1):
                y[i] = b[i]
                for k in range(0, i, 1):
                    y[i] -= y[k] * L[i][k]
                y[i] = y[i] / float(L[i][i])

            # Perform substitution Ux=y
            const = 1
            xl = [0 for i in range(n)]
            for i in range(n - 1, -1, -1):
                xl[i] = y[i]
                for k in range(i + 1, n, 1):
                    xl[i] -= xl[i + const] * U[i][k]
                    const = const + 1
                xl[i] = xl[i] / float(U[i][i])
                const = 1
            write_output(xl)
            end_time = timer()
            time_taken = (end_time - start_time) * 1000
            pprint.pprint(xl)
            output = Label(root, text="LU Decomposition Solution: ", relief="solid", font=("ariel", 9, "bold"))
            output.grid(row=14, column=0)
            exec = Label(root, text="Execution Time taken " + str(time_taken) + " msec", relief="solid",
                         font=("ariel", 9, "bold"))
            exec.grid(row=15, column=1)
            w = Text(root, width=19, height=5)
            w.grid(row=14, column=1)
            w.insert(END, xl)


def Gausse_seidal(aug, n):
    str3 = 'Gauss-Seidel'
    if method.get() == str3 or isall.get() == 'Run All Methods':
        start_time = timer()
        # n,aug,x-->initial points,g-->tol,maxir-->maximum iteration
        a = np.zeros((n, n))  # augemented matrix a initialized by zeros
        b = np.zeros(n)  # augemented matrix a initialized by zeros
        xs =np.zeros(n)
        # for ind in range(n):
        # x[ind] = float(inital_points[ind])

        for i in range(n):
            for j in range(n):
                a[i][j] = aug[i][j]
                # print('a[' + str(i) + '][' + str(j) + ']',a[i][j])

        c = 0
        for i in range(n):
            summ = 0
            f = 0
            dominant = 0
            for j in range(n):
                if i == j:
                    dominant = abs(a[i][j])
                    print('domn', dominant)
                else:
                    summ += abs(a[i][j])
                    print('sum', summ)
            if dominant >= summ:
                if dominant > summ:
                    c += 1
            else:
                f = 1
            if f == 1 and c > n - 1:
                break
        i = 0
        while i < n:
            b[i] = aug[i][n]
            i += 1
        while i < n:
            xs[i] = aug[i][n]
            i += 1
        if maxEntry.index("end") == 0:
            maxitr = 50
            print('maxitr' + str(maxitr))
        else:
            maxitr = float(maxEntry.get())
            print('maxitr' + str(maxitr))
        xs = xs.astype(float)  # Set the precision of x, so that multiple decimals can be displayed in the calculation of x
        m, n = a.shape
        x1 = []
        x2 = []
        x3 = []
        itr = []
        j = 0  # for graph
        while j < n:
            x1.append(xs[j])  # 1st root
            j += 1
            x2.append(xs[j])  # 2nd root
            j += 1
            x3.append(xs[j])
            j += 1

        times = 0  # Iterations
        itr.append(times)
        if f == 0:
            listofiter = []
            listofx = []
            output = Label(root, text=" step ", relief="solid", font=("ariel", 9, "bold"))
            output.grid(row=6, column=2)
            out = Label(root, text=" root ", relief="solid", font=("ariel", 9, "bold"))
            out.grid(row=6, column=3)
            if percEntry.index("end") == 0:
                g = 0.0001
                print('g' + str(g))
            else:
                g = float(percEntry.get())
                print('g' + str(g))
            f = open("demofile3.txt", "w")
            f.write("The roots are:\n")
            print('maxitr before while ' + str(maxitr))
            while times < maxitr:
                for i in range(n):
                    s1 = 0
                    tempx = xs.copy()  # Record the answer of the last iteration
                    for j in range(n):
                        if i != j:
                            s1 += xs[j] * a[i][j]
                    xs[i] = (b[i] - s1) / a[i][i]  # dividing on diagonal elements a11,a22,a33
                j = 0  # for graph
                while j < n:
                    x1.append(xs[j])  # 1st root
                    j += 1
                    x2.append(xs[j])  # 2nd root
                    j += 1
                    x3.append(xs[j])
                    j += 1
                gap = max(abs((xs - tempx) / xs))  # Difference from the last answer
                if gap < g:  # Accuracy meets the requirements, end
                    break
                listofiter.append(Text(root, width=20, height=2))
                listofiter[times].grid(row=7 + times, column=2)
                listofiter[times].insert(END, times + 1)
                listofx.append(Text(root, width=35, height=2))
                listofx[times].grid(row=7 + times, column=3)
                listofx[times].insert(END, xs)
                times += 1
                itr.append(times)
                print(times, xs)
                f.write(str(times))
                f.write(" ")
                for i in range(len(xs)):
                    f.write(str(xs[i]))
                    f.write(" ")
                f.write("\n")
            f.close()
            end_time = timer()
            time_taken = (end_time - start_time)
            exec = Label(root, text="Execution Time taken " + str(time_taken) + " msec", relief="solid",
                         font=("ariel", 9, "bold"))
            exec.grid(row=4, column=2)
            iterNo = Label(root, text="Number of Iterations taken " + str(times), relief="solid",
                           font=("ariel", 9, "bold"))
            iterNo.grid(row=4, column=3)
            itr.append(times + 1)
            plt.plot(itr, x1, label="root1")
            plt.plot(itr, x2, label="root2")
            plt.plot(itr, x3, label="root3")
            plt.legend()
            plt.show()
            plt.xlabel('iterations')
            plt.ylabel('roots')
            print('seidal')
            print(time_taken)


#End Of Functions

NoEq = Label(root, text="Enter the number of equations", relief="solid", font=("ariel", 9, "bold"))
NoEq.grid(row=0)

NoEqEntry = Entry(root, textvar=N)
NoEqEntry.grid(row=0, column=1)

perc = Label(root, text="Enter the percision", relief="solid", font=("ariel", 9, "bold"))
perc.grid(row=1)

percEntry = Entry(root, textvar=percision)
percEntry.grid(row=1, column=1)

maxi = Label(root, text="Enter the max no of iterations", relief="solid", font=("ariel", 9, "bold"))
maxi.grid(row=2)

maxEntry = Entry(root, textvar=maxitr)
maxEntry.grid(row=2, column=1)

methodlist = ['Gaussian-elimination', 'Gaussian-jordan', 'LU decomposition', 'Gauss-Seidel']
droplist = OptionMenu(root, method, *methodlist)
method.set("Select Method")
droplist.config(width=30)
droplist.grid(row=0, column=2)

inputlist = ['Data Entered By User', 'Data Entered From File']
droplist2 = OptionMenu(root, InputType, *inputlist)
InputType.set("Select Input Type")
droplist2.config(width=30)
droplist2.grid(row=1, column=2)

alllist = ['Only One Method', 'Run All Methods']
droplist3 = OptionMenu(root, isall, *alllist)
isall.set("Select Method Type")
droplist3.config(width=30)
droplist3.grid(row=2, column=2)

ok = Button(root, text='Enter Equations', width=25, command=lambda: [user_input()])
ok.grid(row=3, column=0)

okinit = Button(root, text='Enter Initial Points', width=25, command=lambda: [init_input()])
okinit.grid(row=3, column=1)

calculate = Button(root, text='Calculate', width=25,
                   command=lambda: [temp1(), temp2(), fileinput(), decompose(matrix), Gauess_jordan(matrix, numberofrows),
                                    Gauess_Elimination(matrix, numberofrows), Gausse_seidal(matrix, numberofrows)])
calculate.grid(row=3, column=2)

root.mainloop()