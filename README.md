# PPO_LSTM_Car_Navigation
A car-agent leans to navigate in complex traffic conditions by PPO.  
The agent is supposed to learn to choose accelerations to reach the destination avoiding causing jam or collision.  
# Traffic conditions && Collision Detection
When a car-agent navigates on the road, it may encounter with other cars.   
In some conditions, the acceleration chosen by car-agent will cause jam or collision.  
Since the condition will come very complex and the GAMA simulator has no idea about the collision so I have to make collision detection or jam detection.  
These equations are neccessary. And here will use Euclidean distance for safe driving.   
<a href="https://www.codecogs.com/eqnedit.php?latex=S&space;=&space;v_{0}*t&space;&plus;&space;\frac{1}{2}at^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?S&space;=&space;v_{0}*t&space;&plus;&space;\frac{1}{2}at^{2}" title="S = v_{0}*t + \frac{1}{2}at^{2}" /></a>   
v_{n+1} = v_{n}+a_{n}t_{n}

## On the same road
First, the agent compute the useful distances (There will be distance of the behind car or distance of the front car).   
And then detections will be executed after the agent choose acceleration to detecte whether the acceleration will cause jams or collisions.    
A unit of time is 1-cycle.  
### Collision Detection
When there is an another car is in front of the car-agent when the two cars are on the same road, if   
                                      <a href="https://www.codecogs.com/eqnedit.php?latex=EuclideanDistance&space;&plus;&space;v_{car}&space;\leq&space;v_{agent}*t&plus;\frac{1}{2}*a*t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?EuclideanDistance&space;&plus;&space;v_{car}&space;\leq&space;v_{agent}*t&plus;\frac{1}{2}*a*t" title="EuclideanDistance + v_{car} \leq v_{agent}*t+\frac{1}{2}*a*t" /></a>
                                      
### Jam Detection
