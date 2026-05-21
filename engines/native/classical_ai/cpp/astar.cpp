#include <iostream>
#include <vector>
#include <queue>
#include <cmath>

using namespace std;

// Represents a point in a 2D grid
struct Point {
    int x, y;
    bool operator==(const Point& o) const { return x == o.x && y == o.y; }
};

// Node for priority queue
struct Node {
    Point pos;
    int g, h;
    Node(Point p, int g, int h) : pos(p), g(g), h(h) {}
    int f() const { return g + h; }
    bool operator>(const Node& o) const { return f() > o.f(); }
};

// Manhattan distance heuristic
int heuristic(Point a, Point b) {
    return abs(a.x - b.x) + abs(a.y - b.y);
}

// A* Search on a 2D grid (0 = free, 1 = obstacle)
int a_star(const vector<vector<int>>& grid, Point start, Point goal) {
    int rows = grid.size(), cols = grid[0].size();
    priority_queue<Node, vector<Node>, greater<Node>> pq;
    vector<vector<int>> cost(rows, vector<int>(cols, 1e9));

    pq.push(Node(start, 0, heuristic(start, goal)));
    cost[start.x][start.y] = 0;

    int dx[] = {-1, 1, 0, 0};
    int dy[] = {0, 0, -1, 1};

    while (!pq.empty()) {
        Node curr = pq.top();
        pq.pop();

        if (curr.pos == goal) return curr.g;

        for (int i = 0; i < 4; ++i) {
            int nx = curr.pos.x + dx[i];
            int ny = curr.pos.y + dy[i];

            if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && grid[nx][ny] == 0) {
                int new_cost = curr.g + 1;
                if (new_cost < cost[nx][ny]) {
                    cost[nx][ny] = new_cost;
                    pq.push(Node({nx, ny}, new_cost, heuristic({nx, ny}, goal)));
                }
            }
        }
    }
    return -1; // Path not found
}

int main() {
    vector<vector<int>> grid = {
        {0, 0, 0, 0},
        {0, 1, 1, 0},
        {0, 0, 0, 0}
    };
    Point start = {0, 0};
    Point goal = {2, 3};
    cout << "A* Shortest Path Cost: " << a_star(grid, start, goal) << endl;
    return 0;
}
