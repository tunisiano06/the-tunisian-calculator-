import math as ma
import statistics as st
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import sympy as sp
def safe_input(prompt, cast_type=int):
    while True:
        try:
            return cast_type(input(prompt))
        except ValueError:
            print(" Input error, aawed da5el valeur s7i7")


def mathsym():
    func = input(
    ).strip().lower()
    val = safe_input("Da5al el value: ", float)
    if func == "ln":
        result = ma.log(val)
    elif func == "e":
        result = ma.exp(val)
    elif func == "log":
        result = ma.log10(val)
    elif func == "sqrt":
        result = ma.sqrt(val)
    elif func == "cos":
        result = ma.cos(val)
    elif func == "sin":
        result = ma.sin(val)
    elif func == "tan":
        result = ma.tan(val)
    else:
        print("Invalid function")
        return None

    print(f"Result of {func}({val}) = {result}")
    return result
class Cal_trad:
    @staticmethod
    def pls(a, b):
        result = a + b
        print("El natija:", result)
        return result

    @staticmethod
    def subs(a, b):
        result = a - b
        print(" El natija:", result)
        return result

    @staticmethod
    def fois(a, b):
        result = a * b
        print(" El natija:", result)
        return result

    @staticmethod
    def div(a, b):
        if b == 0:
            print(" Division by zero!")
            return float("inf")
        result = a / b
        print("El natija:", result)
        return result


class Cal_Adv:
    @staticmethod
    def fonc(a, b, c):
        oprt = input("Chniya el fonction: [1-affine/2-lineaire/3-quadra] : ").strip().lower()
        print("Optional: apply math function on x")
        apply_math = input("T7eb test3mel mathsym? [y/n]: ").strip().lower()
        if apply_math == "y":
            x = mathsym()
        else:
            x = safe_input("Da5al el x: ", float)

        if x is None:
            return None

        if oprt == "1":
            result = a * x + b
        elif oprt == "2":
            result = a * x
        elif oprt == "3":
            result = a * x ** 2 + b * x + c
        else:
            print(" Mouch fonction ma3roufa")
            return None
        print(f"f(x) = {result}")
        return result
class Other_Func_Cal:
    @staticmethod
    def matrix():
        f = safe_input("Rows: ")
        g = safe_input("Columns: ")
        mat = []
        for i in range(f):
            row = []
            for j in range(g):
                n = safe_input(f"({i},{j}): ", float)
                row.append(n)
            mat.append(row)
        print(" Matrix:")
        print(tabulate(mat, tablefmt="grid"))
        return mat

    @staticmethod
    def statistics():
        x = []
        y = []
        n = safe_input("Nb datapoints: ")
        for i in range(n):
            x.append(safe_input(f"x[{i+1}]: ", float))
            y.append(safe_input(f"y[{i+1}]: ", float))
        print(" Mean x:", st.mean(x))
        print(" Mean y:", st.mean(y))
        if n >= 2:
            print(" Variance x:", st.variance(x))
            print(" Variance y:", st.variance(y))
            print(" Std Dev x:", st.stdev(x))
            print(" Std Dev y:", st.stdev(y))
        cov = np.cov(x, y)[0][1]
        corr = np.corrcoef(x, y)[0][1]
        print(" Covariance:", cov)
        print(" Correlation:", corr)
        c = input("T7eb graphic ? [y/n]: ").lower().strip()
        if c == "y":
            plt.figure()
            plt.scatter(x, y)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Scatter Plot")
            plt.grid(True)
            plt.show()

    @staticmethod
    def suite():
        fnc = input("Type [1-arithmetic/2-geometric]: ").lower().strip()
        n = safe_input("Nb terms: ")
        a = safe_input("First term: ", float)
        d = safe_input("Common diff/ratio: ", float)
        if fnc == "1":
            series = [a + i * d for i in range(n)]
        elif fnc == "2":
            series = [a * (d ** i) for i in range(n)]
        else:
            print(" Fonction mouch mawjouda")
            return []
        print(" Series:", series)
        c = input("T7eb graphic ? [y/n]: ").lower().strip()
        if c == "y":
            plt.figure()
            plt.plot(series, marker="o")
            plt.title("Series Plot")
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.grid(True)
            plt.show()
        return series

    @staticmethod
    def probability():
        choice = input("Type [tirage/tree]: ").strip().lower()
        n = safe_input("Nb trials: ")
        p = safe_input("Probability event: ", float)
        if not (0.0 <= p <= 1.0):
            print(" mabin 0 ou 1 yr7m bok :>")
            return None
        q = 1 - p
        if choice == "tirage":
            result = p ** n
        elif choice == "tree":
            result = p * (q ** (n - 1))
        else:
            print(" Type inconnu")
            return None
        print(f" Probability: {result}")
        return result

    @staticmethod
    def complex():
        real = safe_input("Real part: ", float)
        imag = safe_input("Imaginary part: ", float)
        z = complex(real, imag)
        print(f" Complex number: {z}")
        print(f" Modulus: {abs(z)}")
        print(f" Conjugate: {z.conjugate()}")
        c = input("T7eb graphic [y/n]: ").lower().strip()
        if c == "y":
            plt.figure()
            plt.axhline(0)
            plt.axvline(0)
            plt.plot([0, real], [0, imag], marker="o")
            plt.title("Complex Number on Argand Plane")
            plt.xlabel("Real Part")
            plt.ylabel("Imaginary Part")
            plt.grid(True)
            plt.show()
        return z
    @staticmethod
    def graph_function():
        x = sp.Symbol('x')
        expr_input = input(" Da5al el fonction f(x): ")
        try:
            f = sp.sympify(expr_input)
        except Exception:
            print("Erreur f(x) mouch ma3roufa")
            return
        xmin = safe_input("x min (default -10): ", float)
        xmax = safe_input("x max (default 10): ", float)
        if xmin >= xmax:
            print(" it should be xmin  < xmax")
            return
        fn = sp.lambdify(x, f, modules=["numpy"])
        X = np.linspace(xmin, xmax, 800)
        Y = fn(X)
        Y = np.array(Y, dtype=float)
        plt.figure()
        plt.plot(X, Y)
        plt.title(f"f(x) = {sp.sstr(f)}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.grid(True)
        plt.show()
    @staticmethod
    def variation_table_text():
        x = sp.Symbol('x')
        expr_input = input(" Da5al el fonction f(x): ")
        try:
            f = sp.sympify(expr_input)
        except Exception:
            print(" Erreur f(x) mouch ma3roufa")
            return

        fprime = sp.diff(f, x)
        try:
            crit = list(sp.nroots(fprime)) if fprime.is_polynomial() else sp.solve(sp.Eq(fprime, 0), x)
        except Exception:
            crit = sp.solve(sp.Eq(fprime, 0), x)
        crit = [sp.N(c) for c in crit if c.is_real]
        try:
            sing = list(sp.calculus.util.singularities(f, x, domain=sp.S.Reals))
        except Exception:
            sing = []
        points = sorted(set([sp.N(p) for p in crit + sing if sp.im(p) == 0]))

        # Build intervals
        endpoints = [-sp.oo] + points + [sp.oo]
        rows = []
        header = ["Interval", "Sign f'(x)", "Variation", "Notes"]

        for i in range(len(endpoints) - 1):
            a, b = endpoints[i], endpoints[i+1]
            # pick test point
            if a is -sp.oo:
                t = b - 1
            elif b is sp.oo:
                t = a + 1
            else:
                t = (a + b) / 2
            try:
                val = sp.N(fprime.subs(x, t))
            except Exception:
                val = sp.nan
            sign = "+" if val.is_real and val > 0 else ("-" if val.is_real and val < 0 else "0/NaN")
            arrow = "↑" if sign == "+" else ("↓" if sign == "-" else "?")
            rows.append([f"({sp.sstr(a)}, {sp.sstr(b)})", sign, arrow, ""])
        extrema_notes = []
        for p in points:
            if p in sing:
                extrema_notes.append([sp.sstr(p), "—", "singularity/asymptote"])
                continue
            try:
                fval = sp.N(f.subs(x, p))
            except Exception:
                fval = sp.nan
            left_sign = None
            right_sign = None
            for i in range(len(endpoints) - 1):
                a, b = endpoints[i], endpoints[i+1]
                if a < p < b:
                    if i-1 >= 0:
                        la, lb = endpoints[i-1], endpoints[i]
                        lt = (la + lb)/2 if la is not -sp.oo and lb is not sp.oo else (lb - 1 if la is -sp.oo else la + 1)
                        try:
                            lv = sp.N(fprime.subs(x, lt))
                            left_sign = 1 if lv > 0 else (-1 if lv < 0 else 0)
                        except Exception:
                            left_sign = None
                    ra, rb = endpoints[i], endpoints[i+1]
                    rt = (ra + rb)/2 if ra is not -sp.oo and rb is not sp.oo else (rb - 1 if ra is -sp.oo else ra + 1)
                    try:
                        rv = sp.N(fprime.subs(x, rt))
                        right_sign = 1 if rv > 0 else (-1 if rv < 0 else 0)
                    except Exception:
                        right_sign = None
                    break
            nature = "—"
            if left_sign is not None and right_sign is not None:
                if left_sign > 0 and right_sign < 0:
                    nature = "local max"
                elif left_sign < 0 and right_sign > 0:
                    nature = "local min"
                elif left_sign == right_sign:
                    nature = "saddle/flat"
            extrema_notes.append([sp.sstr(p), sp.sstr(fval), nature])

        print("\n f(x) =", sp.sstr(f))
        print(" f'(x) =", sp.sstr(fprime))
        print("\n=== Tableau de variation (Intervals) ===")
        print(tabulate(rows, headers=header, tablefmt="grid"))
        if extrema_notes:
            print("\n===  Extrema & Points ===")
            print(tabulate(extrema_notes, headers=["x", "f(x)", "Type"], tablefmt="grid"))
    @staticmethod
    def variation_table_plot():
        x = sp.Symbol('x')
        expr_input = input(" Da5al el fonction f(x): ")
        try:
            f = sp.sympify(expr_input)
        except Exception:
            print("Erreur f(x) mouch ma3roufa")
            return
        fprime = sp.diff(f, x)
        try:
            crit = sp.solve(sp.Eq(fprime, 0), x)
        except Exception:
            crit = []
        crit = [sp.N(c) for c in crit if c.is_real]
        try:
            sing = list(sp.calculus.util.singularities(f, x, domain=sp.S.Reals))
        except Exception:
            sing = []
        points = sorted(set([sp.N(p) for p in crit + sing if sp.im(p) == 0]))

        xmin = safe_input("x min (default -10): ", float)
        xmax = safe_input("x max (default 10): ", float)
        if xmin >= xmax:
            print("  rod balek : xmin  < xmax")
            return
        fn = sp.lambdify(x, f, modules=["numpy"])  
        X = np.linspace(xmin, xmax, 1200)
        Y = fn(X)
        Y = np.array(Y, dtype=float)

        plt.figure()
        plt.plot(X, Y, label=f"f(x) = {sp.sstr(f)}")
        for p in points:
            try:
                yv = float(sp.N(f.subs(x, p)))
                plt.scatter([float(p)], [yv])
                plt.annotate(f"x={sp.sstr(p)}\n f={yv:.3g}", (float(p), yv))
            except Exception:
                pass
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Tableau de variation — Graphique")
        plt.grid(True)
        plt.legend()
        plt.show() 
    def factorial():
        x = int(input("da5al el number x : "))
        n=ma.factorial(x)
        print(f"the result is  {n}")
def choice():
    while True:
        print("\n=== 🖩 MAIN MENU ===")
        print("[1] - Traditional Calculator")
        print("[2] - Advanced Calculator")
        print("[3] - Other Functions")
        print("[4] - Exit")
        ch = safe_input("Choose: ")

        if ch == 1:
            print("\n[1]-Addition\n[2]-Substraction\n[3]-Multiplication\n[4]-Division")
            op = safe_input("i5tar: ")
            a = safe_input("a: ", float)
            b = safe_input("b: ", float)
            if op == 1:
                Cal_trad.pls(a, b)
            elif op == 2:
                Cal_trad.subs(a, b)
            elif op == 3:
                Cal_trad.fois(a, b)
            elif op == 4:
                Cal_trad.div(a, b)
            else:
                print(" Wrong choice")

        elif ch == 2:
            a = safe_input("a: ", float)
            b = safe_input("b: ", float)
            c = safe_input("c: ", float)
            Cal_Adv.fonc(a, b, c)

        elif ch == 3:
            print("\n [1]- Matrix")
            print("[2]- Statistics")
            print("[3]- Suite")
            print("[4]- Probability")
            print("[5]- Complex")
            print("[6]- Graph f(x)")
            print("[7]- Tableau de variation (Text)")
            print("[8]- Tableau de variation (Graph)")
            print("[9]- factorial")
            opt = safe_input("i5tar: ")
            if opt == 1:
                Other_Func_Cal.matrix()
            elif opt == 2:
                Other_Func_Cal.statistics()
            elif opt == 3:
                Other_Func_Cal.suite()
            elif opt == 4:
                Other_Func_Cal.probability()
            elif opt == 5:
                Other_Func_Cal.complex()
            elif opt == 6:
                Other_Func_Cal.graph_function()
            elif opt == 7:
                Other_Func_Cal.variation_table_text()
            elif opt == 8:
                Other_Func_Cal.variation_table_plot()
            elif opt ==9:
                Other_Func_Cal.factorial()
            else:
                print(" Wrong choice")

        elif ch == 4:
            print(" Inshallah zina :)")
            break
        else:
            print(" Wrong choice")


if __name__ == "__main__":
    choice()
