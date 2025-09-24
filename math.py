import itertools, math

def search_expressions(numbers, target, tol=1e-6):
    ops = ['+', '-', '*', '/']
    solutions = []
    
    def eval_safe(expr):
        try:
            val = eval(expr)
            if isinstance(val, (int, float)) and math.isfinite(val):
                return val
        except ZeroDivisionError:
            return None
        except OverflowError:
            return None
        return None

    for perm in itertools.permutations(numbers):
        a, b, c = perm
        for op1 in ops:
            for op2 in ops:
                exprs = [
                    f"({a}{op1}{b}){op2}{c}",
                    f"{a}{op1}({b}{op2}{c})"
                ]
                for expr in exprs:
                    val = eval_safe(expr)
                    if val is None:
                        continue
                    diff = abs(val - target)
                    if diff <= tol:
                        solutions.append((expr, val, diff))
                    elif not solutions or diff < min(s[2] for s in solutions):
                        solutions.append((expr, val, diff))
    # 按接近程度排序
    solutions.sort(key=lambda x: x[2])
    return solutions[:10]

if __name__ == "__main__":
    # 手动输入数据
    nums_str = input("请输入数字（用逗号分隔，例如 58,53,128）：")
    numbers = [int(x.strip()) for x in nums_str.split(",")]
    target = float(input("请输入目标值："))

    results = search_expressions(numbers, target)
    print("\n最接近目标的算式：")
    for expr, val, diff in results:
        print(f"{expr} = {val} (diff={diff})")
