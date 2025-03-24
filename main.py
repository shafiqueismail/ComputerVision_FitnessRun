import pygame
import os
import json
import random

pygame.init()

screen = pygame.display.set_mode((0, 0), pygame.RESIZABLE)

pygame.display.set_caption('Project Run')

# set fixed framerate
clock = pygame.time.Clock()
FPS = 60

# define game variables
GRAVITY = 0.75
SCREEN_WIDTH = pygame.display.get_surface().get_size()[0] 
SCREEN_HEIGHT = pygame.display.get_surface().get_size()[1] #this gets the hight, if the screen gets resized maybe we shold reinitialize this
SCROLL_DISTANCE_TRESH_LEFT = 200
SCROLL_DISTANCE_TRESH_RIGHT = SCREEN_WIDTH // 2
CHUNK_ROWS = 27
CHUNK_COLS = 49
TILE_SIZE = SCREEN_HEIGHT // CHUNK_ROWS
NUM_TILE_TYPES = 79
world = [] # list of chunks
screen_scroll = 0
scroll_offset = 0


# define player actions variables
moving_left = False
moving_right = False

# load BG images
bg_images = []
bg_images_paths = [
    'Images/Backgrounds_Sky.png',
    'Images/Backgrounds_BuildingsBack.png',
    'Images/Backgrounds_BuildingsMid.png',
    'Images/Backgrounds_BuildingsClose.png',
]
for path in bg_images_paths:

    img = pygame.image.load(path).convert_alpha()
    img = pygame.transform.scale(
        img, 
        (
            int(img.get_width() * (SCREEN_HEIGHT / img.get_height())),
            SCREEN_HEIGHT, 
        )
    )
    bg_images.append(img)

# load tile images
def get_image(sheet, cols, width, height, color, image_index):
    image = pygame.Surface((width, height)).convert_alpha() # this creates a blank image, we will use this as a window to get part of our spread sheet
    # like screen because its also a surface
    image.blit(sheet, (0,0), (image_index % cols * width, image_index // cols * height, width, height))
    image = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
    image.set_colorkey(color) # makes this color transparent?
    return image

BLACK = (0,0,0)

chunk_spritesheet = pygame.image.load('Chunks/1Bit_Platformer/spritesheet.png').convert_alpha()
tile_img_list = []
for i in range(NUM_TILE_TYPES):
    tile_img_list.append(get_image(chunk_spritesheet, 8, 16, 16, BLACK, i))

BG_COLOR = (144, 201, 120)
# BG_COLOR = (0, 0, 0)
LINE_COLOR = (255, 255, 255)

def draw_bg(number):
    screen.fill(BG_COLOR)
    width = bg_images[0].get_width()
    for i in range(number):
        for parallax_i, img in enumerate(bg_images): # should already be in proper order
            parallax_bias = (parallax_i + 5) / 10
            screen.blit(img, (width * i + scroll_offset * parallax_bias, 0))

class Runner(pygame.sprite.Sprite):
    # type is either: player or chaser
    def __init__(self, runner_type, x, y, scale, speed):
        pygame.sprite.Sprite.__init__(self)
        self.runner_type = runner_type
        self.alive = True
        self.speed = speed 
        self.vel_y = 0 # velocity
        self.direction = 1 # 1 is right, -1 is left
        self.flip = False 
        self.jump = False
        self.in_air = True
        self.jump_type = 0 # 0 rise, 1 mid, 2 fall

        # animation
        self.animation_list = []
        self.animation_index = 0
        self.action = 0 # 0 is idle, 1 is run
        self.update_time = pygame.time.get_ticks()

        # load all animations for runner
        animation_types = ['Idle', 'Run', 'JumpRise', 'JumpMid', 'JumpFall']
        for animation in animation_types:
            temp_animation = []
            num_of_frames = len(os.listdir(f'Images/Player/{animation}'))
            for i in range(num_of_frames):
                img = pygame.image.load(f'Images/Player/{animation}/{animation}0{i+1}.png')
                img = pygame.transform.scale(img, (int(img.get_width() * scale), int(img.get_height() * scale)))
                temp_animation.append(img)
            self.animation_list.append(temp_animation)
        
        self.image = self.animation_list[self.action][self.animation_index]
        self.rect = self.image.get_rect() # its not the image thats important its the rectangle
        self.rect.center = (x, y)
        self.rect.width = self.image.get_width() // 6
        self.rect.height = self.image.get_height() // 2
    
    def move(self, moving_left, moving_right):
        # reset movement valriables
        screen_scroll = 0
        dx = 0 # represents the change in x
        dy = 0

        if moving_left:
            dx = -self.speed
            self.flip = True
            self.direction = -1
        if moving_right:
            dx = self.speed
            self.flip = False
            self.direction = 1

        # jump
        if self.jump == True and self.in_air == False:
            self.vel_y = -20 # negative goes up
            self.jump = False
            self.in_air = True
            self.jump_type = 0 # jump rise

        # add gravity
        self.vel_y += GRAVITY
        if self.vel_y > -5: 
            self.jump_type = 1 # jump mid
        if self.vel_y > 5: 
            self.jump_type = 2 # jump fall
        if self.vel_y > 10: # so its not too fast, can remove this later
            self.vel_y = 10 
        dy += self.vel_y

        # collision detection
        for chunk in world:
            for tile in chunk.obstacle_list:
                # x direction
                # the dx here is for future direction
                if tile[1].colliderect(self.rect.x + dx, self.rect.y, self.rect.width, self.rect.height):
                    dx = 0
                # y direction
                # print(self.rect.x, self.rect.y + dy, self.rect.width, self.rect.height)
                if tile[1].colliderect(self.rect.x, self.rect.y + dy, self.rect.width, self.rect.height):
                    if self.vel_y < 0: # jumping
                        self.vel_y = 0
                        dy = tile[1].bottom - self.rect.top
                    elif self.vel_y >= 0: # falling
                        self.vel_y = 0
                        dy = tile[1].top - self.rect.bottom - 1
                        self.in_air = False
        
        # update rect
        self.rect.x += dx
        self.rect.y += dy

        # update scroll
        if self.runner_type == 'player':
            if self.rect.right > SCREEN_WIDTH - SCROLL_DISTANCE_TRESH_RIGHT or (self.rect.left < SCROLL_DISTANCE_TRESH_LEFT and scroll_offset < 0): # < 0 is to stop scrolling on left edge
                self.rect.x -= dx
                screen_scroll = -dx
        
        return screen_scroll

    def update_animation(self):
        ANIMATION_COOLDOWN = 100
        # update image
        self.image = self.animation_list[self.action][self.animation_index % len(self.animation_list[self.action])]
        #check if enough time has passed since last update
        if pygame.time.get_ticks() - self.update_time > ANIMATION_COOLDOWN:
            self.animation_index += 1
            self.update_time = pygame.time.get_ticks()
            
    def update_action(self, new_action):
        # check if new action is different to current
        if new_action != self.action:
            self.action = new_action
            # reset animation settings
            self.animation_index = 0
            self.update_time = pygame.time.get_ticks()

    def draw(self):
        img_rect = self.image.get_rect()
        img_rect.midbottom = self.rect.midbottom

        # # debugging
        # pygame.draw.rect(screen, (0, 255, 0), img_rect)
        # pygame.draw.rect(screen, (0, 0, 255), self.rect)

        screen.blit(
            pygame.transform.flip(self.image, self.flip, False), 
            img_rect
        ) #daw image
        

class Chunk():
    def __init__(self, index):
        self.obstacle_list = []
        self.index = index
        self.x = index * TILE_SIZE * CHUNK_COLS + scroll_offset
    
    def process_data(self, data):
        for y, row in enumerate(data):
            for x, (tile_id, is_collider) in enumerate(row):
                if tile_id >= 0: # ignore -1
                    img = tile_img_list[tile_id]
                    img_rect = img.get_rect()
                    img_rect.x = x * TILE_SIZE
                    img_rect.x += self.index * TILE_SIZE * CHUNK_COLS + scroll_offset
                    img_rect.y = y * TILE_SIZE
                    tile_data = (img, img_rect)
                    if is_collider:
                        self.obstacle_list.append(tile_data)
                    else:
                        pass#non colliders
                    # if you add enemies or powerups just add special tiles that with special layers that says wehre they are and spawn them

    def draw(self):
        self.x += screen_scroll
        for tile in self.obstacle_list:
            # copy_rect = tile[1].copy()
            tile[1].x += screen_scroll
            screen.blit(tile[0], tile[1])



player = Runner('player', 10 * TILE_SIZE, 11 * TILE_SIZE, 3, 10)

chunk_datas = []

# create empty chunk
chunk_data = []
for row in range(CHUNK_ROWS):
    r = [-1] * CHUNK_COLS
    chunk_data.append(r)

# load chunk
with open('Chunks/1Bit_Platformer/map.json') as file:
    data = json.load(file)
    for layer in data['layers']:
        for tile in layer['tiles']:
            x = tile['x']
            y = tile['y']
            if x < CHUNK_COLS and y < CHUNK_ROWS and chunk_data[y][x] == -1: # IMPORTANT: it seems first layer is closer to the screen so this works
                chunk_data[y][x] = (int(tile['id']), layer['collider'])

chunk_datas.append(chunk_data)

# for row in chunk_data:
#     nums = []
#     for num in row:
#         nums.append(num[0])
#     print(nums)


chunk0 = Chunk(0)
chunk0.process_data(chunk_datas[0])
world.append(chunk0) # always start with the first chunk, maybe add a starter chunk later

last_chunk_index = 0
def add_chunks(num, last_chunk_index):
    index = last_chunk_index
    for _ in range(num): # higher number more chunks
        index += 1
        chunk = Chunk(index)
        chunk_data = random.choice(chunk_datas)
        chunk.process_data(chunk_data)
        world.append(chunk) # randomely populate
    print(index)
    return index
    
last_chunk_index = add_chunks(1, last_chunk_index)

num_backgrounds = 2 # this is not number of layers, but number of backgrounds side by side

is_game_running = True
while is_game_running:

    # update background
    if -scroll_offset > bg_images[0].get_width() * (num_backgrounds - 2):
        num_backgrounds += 1
    draw_bg(num_backgrounds) # used to clear the screen

    clock.tick(FPS) # I think it waits to make sure there is a max of 60 frames per second

    if player.rect.left > world[-1].x:
        last_chunk_index = add_chunks(1, last_chunk_index)

    # draw chunks
    for chunk in world:
        chunk.draw()


    player.update_animation()
    player.draw()

    if player.alive:
        #update player actions
        if player.in_air:
            player.update_action(2 + player.jump_type) # jump animation
        elif moving_left or moving_right:
            player.update_action(1) # run animation
        else:
            player.update_action(0) # idle

    screen_scroll = player.move(moving_left, moving_right)
    scroll_offset += screen_scroll

    for event in pygame.event.get():
        # quit game
        if event.type == pygame.QUIT:
            is_game_running = False
        
        # keyboard press
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                moving_left = True
            if event.key == pygame.K_d:
                moving_right = True
            if event.key == pygame.K_w and player.alive:
                player.jump = True
            if event.key == pygame.K_ESCAPE:
                is_game_running = False
        # keyboard unpress
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_a:
                moving_left = False
            if event.key == pygame.K_d:
                moving_right = False
    pygame.display.update() # update the screen

pygame.quit()