import json
from djitellopy import Tello

myDrone = Tello()
myDrone.connect()

# Read datas from JSON file.
f = open('waypoints.json','r')
path_dict = json.load(f)
path_wp = path_dict['wp']
path_pos = path_dict['pos']

print(path_wp)
print(path_pos)


# Follow the instructions
myDrone.takeoff()

for instruction in path_wp:
    angle = instruction['angle_deg']
    length = instruction['dist_cm']
    if (angle < 0):
        myDrone.rotate_counter_clockwise(-angle)
    else:
        myDrone.rotate_clockwise(angle)
    myDrone.move_forward(length)

myDrone.land()

# d.rotate_clockwise(90)
# d.move_forward(20)
# d.land()