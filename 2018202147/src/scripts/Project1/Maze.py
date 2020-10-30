# coding:utf-8
# MazeFindPath(grid): 传入矩阵，返回所有的通路
# 参数: 
#     grid: 传入的迷宫矩阵
#     类型: list(list) [ [] ]
#     每个列表元素表示一行
#     列表元素内某元素为 1 表示起点，2 表示终点，0 表示通路，-1 表示障碍
# 返回值:
#     paths: 所有从起点到终点的通路
#
def MazeFindPath(grid):
    # dfs(r, c): 从 r, c 开始寻路
    def dfs(r,c):
        # 到达终点
        if grid[r][c] == 2:
            # 增加一条通路
            paths.append(path[:])
            return
        # 标记已经访问
        visited[r][c] = True
        # 两个分量的两种变化
        directions = (-1, 1)
        # x分量变化
        for i in directions:
            # 产生新坐标
            x, y = r + i, c
            # 坐标合法，且未访问过，且不是障碍
            if (0 <= x < m) and (0 <= y < n) and (not visited[x][y]) and (grid[x][y] != -1):
                # 本条道路增加一个坐标节点元组
                path.append((x, y))
                # 在新坐标上搜索
                dfs(x, y)
                # 搜索完，弹出坐标
                path.pop()
        # y分量变化        
        for j in directions:
            # 产生新坐标
            x, y = r, c + j
            # 坐标合法，且未访问过，且不是障碍
            if (0 <= x < m) and (0 <= y < n) and (not visited[x][y]) and (grid[x][y] != -1):
                # 本条道路上增加一个元组
                path.append((x, y))
                # 继续找
                dfs(x, y)
                # 弹出
                path.pop()
        # 四个方向都找完了，标记没访问过
        visited[r][c] = False
    # 一条道路
    path = []
    # 返回值: 所有道路
    paths = []
    # 行数、列数
    m, n = len(grid), len(grid[0])
    # 生成访问矩阵
    visited = [ [False] * n for _ in range(m) ]
    # 寻找起点坐标
    start_x, start_y = -1, -1
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                start_x, start_y = i, j
    # 增加起点
    path.append((start_x, start_y))
    # 开始找路
    dfs(start_x, start_y)
    #print(paths)
    return paths


if __name__ == '__main__':
    grid = [
            [ 1, 0, 0, 0,-1],
            [-1, 0,-1, 0,-1],
            [-1, 0,-1, 0, 2],
            [ 0, 0, 0, 0,-1]
        ]
    paths = MazeFindPath(grid)
    path = paths[0]
    print(path)
    delta_path = [(path[j][0] - path[j - 1][0], path[j][1] - path[j - 1][1]) for j in range(1, len(path))]
    print(delta_path)