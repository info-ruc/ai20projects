import pygame
import json
import math
import numpy as np

"""
how many pixel = actul distance in cm
"""
# MAP_SIZE_COEFF = 85.71
MAP_SIZE_COEFF = 1.0

pygame.init()
screen = pygame.display.set_mode([700, 500])
screen.fill((255, 255, 255))
running = True

class Background(pygame.sprite.Sprite):
    def __init__(self, image, location, scale):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(image)
        self.image = pygame.transform.rotozoom(self.image, 0, scale)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

def get_dist_btw_pos(pos0, pos1):
    """
    Get distance between 2 mouse position.
    """
    x = abs(pos0[0] - pos1[0])
    y = abs(pos0[1] - pos1[1])
    dist_px = math.hypot(x, y)
    dist_cm = dist_px * MAP_SIZE_COEFF
    return int(dist_cm), int(dist_px)

def get_angle_btw_line(pos0, pos1, posref):
    """
    Get angle between two lines respective to 'posref'
    NOTE: using dot inner and outer product calculation.
    """
    ax = posref[0] - pos0[0]
    ay = posref[1] - pos0[1]
    bx = pos1[0] - posref[0]
    by = pos1[1] - posref[1]
    v1 = (ax, ay)
    v2 = (bx, by)
    # 2个向量模的乘积
    TheNorm = np.linalg.norm(v1)*np.linalg.norm(v2)
    # 叉乘
    rho =  np.rad2deg(np.arcsin(np.cross(v1, v2)/TheNorm))
    # 点乘
    theta = np.rad2deg(np.arccos(np.dot(v1,v2)/TheNorm))
    # if rho < 0, then rotate counterclockwise 
    if rho < 0:
        return - int(theta)
    # else, then rotate clockwise
    else:
        return int(theta)

"""
Main capturing mouse program.
"""
# load background image.
bground = Background('image.png', [0, 0], 0.8)
screen.blit(bground.image, bground.rect)

path_wp = []
index = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            path_wp.append(pos)
            if index > 0:
                pygame.draw.line(screen, (255, 0, 0), path_wp[index - 1], pos, 2)
            index += 1
    pygame.display.update()

"""
Compute the waypoints (distance and angle).
"""
# Append first pos ref. (dummy)
path_wp.insert(0, (path_wp[0][0], path_wp[0][1] - 10))

path_dist_cm = []
path_dist_px = []
path_angle = []
for index in range(len(path_wp)):
    # Skip the first and second index.
    if index > 1:
        dist_cm, dist_px = get_dist_btw_pos(path_wp[index - 1], path_wp[index])
        path_dist_cm.append(dist_cm)
        path_dist_px.append(dist_px)
    
    # Skip the first and last index.
    if index > 0 and index < (len(path_wp) - 1):
        angle = get_angle_btw_line(path_wp[index-1], path_wp[index+1], path_wp[index])
        path_angle.append(angle)

# Print out the information.
print('path_wp: {}'.format(path_wp))
print('dist_cm: {}'.format(path_dist_cm))
print('dist_px: {}'.format(path_dist_px))
print('dist_angle: {}'.format(path_angle))

"""
Save waypoints into JSON file.
"""
waypoints = []
for index in range(len(path_dist_cm)):
    waypoints.append({
        "dist_cm": path_dist_cm[index],
        "dist_px": path_dist_px[index],
        "angle_deg": path_angle[index]
    })

#Save to JSON file.
f = open("waypoints.json", 'w+')
path_wp.pop(0)
json.dump({
    "wp": waypoints,
    "pos": path_wp
}, f, indent=4)
f.close()