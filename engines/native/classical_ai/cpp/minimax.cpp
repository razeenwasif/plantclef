#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Minimax with Alpha-Beta Pruning
int minimax(int depth, int nodeIndex, bool isMax, vector<int>& scores, int alpha, int beta) {
    if (depth == 3) { // Leaf node (max depth = 3 for a full binary tree of 8 leaves)
        return scores[nodeIndex];
    }

    if (isMax) {
        int best = -1e9;
        for (int i = 0; i < 2; i++) {
            int val = minimax(depth + 1, nodeIndex * 2 + i, false, scores, alpha, beta);
            best = max(best, val);
            alpha = max(alpha, best);
            if (beta <= alpha) break; // Alpha-Beta Pruning
        }
        return best;
    } else {
        int best = 1e9;
        for (int i = 0; i < 2; i++) {
            int val = minimax(depth + 1, nodeIndex * 2 + i, true, scores, alpha, beta);
            best = min(best, val);
            beta = min(beta, best);
            if (beta <= alpha) break; // Alpha-Beta Pruning
        }
        return best;
    }
}

int main() {
    vector<int> scores = {3, 5, 6, 9, 1, 2, 0, -1}; // Leaf nodes
    cout << "Optimal Value (Alpha-Beta Pruning): " << minimax(0, 0, true, scores, -1e9, 1e9) << endl;
    return 0;
}
