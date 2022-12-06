# Life Long Planning Algorithm

## Commands to run the Project

Our project github repository: https://github.com/dobulare/dstar

Please find the final report and team effectiveness report under the https://github.com/dobulare/dstar/report

dstar/dstarlite has project code files
dstar/report has the detailed report of all the search algorithms used in the project.

Prerequisites:
Install Anaconda using this https://docs.anaconda.com/anaconda/install/

Anaconda environment:
```
$ conda create --name cse571 python=3.6
$ source activate cse571
```

Executing the code:
```
$ cd dstarlite
$ python pacman.py -l mediumMaze -p SearchAgent -a fn=dstar,heuristic=manhattanHeuristic --frameTime 0
```

To run the code in different layouts we can replace mediumMaze with any of the following 
layout names =  custom2022 | custom3035 |  bigMaze | tinyMaze | mediumMaze | smallMaze

To use different search functions you can replace the dstar to any of the following function names:
fn = dstar | rastar | rbfs | rdfs | rucs
rastar = replanningAStar
rbfs = replanningBFS
rdfs = replanningDFS
rucs = replanningUCS
dstar = dStarLiteSearch
