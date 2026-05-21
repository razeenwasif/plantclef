module Minimax where

-- A generalized Minimax algorithm implementation in Haskell

data Player = Maximizer | Minimizer deriving (Eq, Show)

-- A generic Game Tree
data GameTree a = Leaf a | Node Player [GameTree a] deriving (Show)

-- Minimax evaluation of a game tree
minimax :: Ord a => GameTree a -> a
minimax (Leaf val) = val
minimax (Node Maximizer children) = maximum (map minimax children)
minimax (Node Minimizer children) = minimum (map minimax children)

-- Example usage
exampleTree :: GameTree Int
exampleTree = Node Maximizer [
                Node Minimizer [Leaf 3, Leaf 5],
                Node Minimizer [Leaf 6, Leaf 9],
                Node Minimizer [Leaf 1, Leaf 2]
              ]

main :: IO ()
main = do
    putStrLn "Evaluating Game Tree using Minimax:"
    print $ minimax exampleTree
