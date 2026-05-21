module KMeans where

import Data.List (minimumBy, groupBy, sortOn)
import Data.Function (on)

type Point = (Double, Double)
type Cluster = [Point]

-- Calculate Euclidean distance
distance :: Point -> Point -> Double
distance (x1, y1) (x2, y2) = sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Find the closest centroid to a point
closestCentroid :: [Point] -> Point -> Point
closestCentroid centroids p = minimumBy (compare `on` distance p) centroids

-- Group points by their closest centroid
assignClusters :: [Point] -> [Point] -> [[Point]]
assignClusters centroids points = 
    let pairs = [(closestCentroid centroids p, p) | p <- points]
        sortedPairs = sortOn fst pairs
        grouped = groupBy ((==) `on` fst) sortedPairs
    in map (map snd) grouped

-- Calculate the center of a cluster
calculateCentroid :: Cluster -> Point
calculateCentroid cluster = 
    let n = fromIntegral (length cluster)
        sumX = sum (map fst cluster)
        sumY = sum (map snd cluster)
    in (sumX / n, sumY / n)

-- One step of k-means
kMeansStep :: [Point] -> [Point] -> [Point]
kMeansStep centroids points = map calculateCentroid (assignClusters centroids points)

main :: IO ()
main = do
    let points = [(1.0, 1.0), (1.5, 2.0), (3.0, 4.0), (5.0, 7.0), (3.5, 5.0), (4.5, 5.0), (3.5, 4.5)]
    let initialCentroids = [(1.0, 1.0), (5.0, 7.0)]
    
    putStrLn "Initial Centroids:"
    print initialCentroids
    
    let newCentroids = kMeansStep initialCentroids points
    putStrLn "Centroids after 1 step:"
    print newCentroids
