import pygame
from pygame import mixer
import os
import json
import random
from new_squat_counter import squat_detector
import multiprocessing
import numpy as np
# from squat_counter import SquatCounter
# import numpy as np
# import multiprocessing
# import cv2

if __name__ == '__main__':
    multiprocessing.freeze_support() 

    mixer.init()
    pygame.init()

    # pygame.mixer.Sound('')

    screen = pygame.display.set_mode((0, 0), pygame.RESIZABLE)

    pygame.display.set_caption('Project Run')

    # set fixed framerate
    clock = pygame.time.Clock()
    FPS = 60

    # define game variables
    GRAVITY = 0.75
    SCREEN_WIDTH = pygame.display.get_surface().get_size()[0] 
    SCREEN_HEIGHT = pygame.display.get_surface().get_size()[1] #this gets the hight, if the screen gets resized maybe we should reinitialize this
    SCROLL_DISTANCE_TRESH_LEFT = 200
    SCROLL_DISTANCE_TRESH_RIGHT = SCREEN_WIDTH // 2
    CHUNK_ROWS = 27
    CHUNK_COLS = 49
    FONT = pygame.font.SysFont('Consolas', 40)
    TILE_SIZE = SCREEN_HEIGHT // CHUNK_ROWS
    NUM_TILE_TYPES = 79
    world = [] # list of chunks
    screen_scroll = 0
    scroll_offset = 0
    game_start = False
    score = 0
    high_score = 0

    # colors
    BLACK = (0,0,0)
    WHITE = (255,255,255)
    BG_COLOR = (144, 201, 120)
    GREY = (213, 220, 227)
    # BG_COLOR = (0, 0, 0)
    LINE_COLOR = (255, 255, 255)

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


    def draw_text(text, font, text_color, x, y):
        img = font.render(text, True, text_color)
        screen.blit(img, (x,y))

    def draw_text_with_outline(text, font, text_color, outline_color, x, y,):
        thickness = 3
        draw_text(text, font, outline_color, x - thickness, y)
        draw_text(text, font, outline_color, x, y - thickness)
        draw_text(text, font, outline_color, x - thickness, y - thickness)
        draw_text(text, font, outline_color, x + thickness, y)
        draw_text(text, font, outline_color, x, y + thickness)
        draw_text(text, font, outline_color, x + thickness, y + thickness)
        draw_text(text, font, outline_color, x + thickness, y - thickness)
        draw_text(text, font, outline_color, x - thickness, y + thickness)
        draw_text(text, font, text_color, x, y)

    def draw_score():
        draw_text_with_outline('Score ' + str(score), FONT, GREY, BLACK, 10, 10)
        draw_text_with_outline('Best ' + str(high_score), FONT, GREY, BLACK, 10, 60)

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
            self.pause_animation = False

            # animation
            self.animation_list = []
            self.animation_index = 0
            self.action = 0 # 0 is idle, 1 is run
            self.update_time = pygame.time.get_ticks()

            # load all animations for runner
            animation_types = ['Idle', 'Run', 'JumpRise', 'JumpMid', 'JumpFall', 'Knockback']
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
        
        def update(self):
            self.update_animation()
            self.eliminate_if_applicable()

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
            if self.alive:
                for chunk in world:
                    for tile in chunk.obstacle_list:
                        # x direction
                        # the dx here is for future direction
                        if tile[1].colliderect(self.rect.x + dx, self.rect.y, self.rect.width, self.rect.height):
                            dx = 0
                            # if self.runner_type == 'player':
                            #     self.alive = False
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

            if self.rect.bottom + 10 > SCREEN_HEIGHT:
                self.alive = False

            # don't go of the left edge
            if self.rect.left + dx <= 0:
                dx = 0
            
            # update rect
            self.rect.x += dx
            self.rect.y += dy

            # update scroll
            if self.runner_type == 'player':
                if self.rect.right > SCREEN_WIDTH - SCROLL_DISTANCE_TRESH_RIGHT \
                    or (self.rect.left < SCROLL_DISTANCE_TRESH_LEFT and scroll_offset < 0): # < 0 is to stop scrolling on left edge
                    self.rect.x -= dx
                    screen_scroll = -dx
            
            return screen_scroll

        def update_animation(self):
            ANIMATION_COOLDOWN = 100
            # update image
            self.image = self.animation_list[self.action][self.animation_index % len(self.animation_list[self.action])]
            #check if enough time has passed since last update
            if pygame.time.get_ticks() - self.update_time > ANIMATION_COOLDOWN:
                if not self.pause_animation or (self.animation_index % len(self.animation_list[self.action]) != len(self.animation_list[self.action]) - 1):
                    self.animation_index += 1
                    self.update_time = pygame.time.get_ticks()
                
        def update_action(self, new_action):
            # check if new action is different to current
            if new_action != self.action:
                self.action = new_action
                # reset animation settings
                self.animation_index = 0
                self.update_time = pygame.time.get_ticks()
        
        def eliminate_if_applicable(self):
            if not self.alive:
                self.speed = 0
                self.update_action(5)
                self.pause_animation = True
            return self.alive

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
        
        def process_data(self, data, tile_img_list):
            for y, row in enumerate(data):
                for x, (tile_id, is_collider) in enumerate(row):
                    if tile_id >= 0 and tile_id < len(tile_img_list): # ignore -1
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

    main_menu_img = pygame.image.load('Images/MainMenu.png').convert_alpha()
    main_menu_rect = main_menu_img.get_rect()
    main_menu_rect.center = screen.get_rect().center

    restart_menu_img = pygame.image.load('Images/RestartMenu.png').convert_alpha()
    restart_menu_rect = restart_menu_img.get_rect()
    restart_menu_rect.center = screen.get_rect().center

    # get highscore
    if os.path.exists('score.txt'):
        with open('score.txt', 'r') as file:
            try:
                high_score = int(file.read())
            except ValueError:
                high_score = 0
    else:
        high_score = 0


    # load tile images
    def get_image(sheet, cols, width, height, color, image_index):
        image = pygame.Surface((width, height)).convert_alpha() # this creates a blank image, we will use this as a window to get part of our spread sheet
        # like screen because its also a surface
        image.blit(sheet, (0,0), (image_index % cols * width, image_index // cols * height, width, height))
        image = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
        image.set_colorkey(color) # makes this color transparent?
        return image

    chunk_datas = []
    num_chunks = len(os.listdir('Chunks/'))
    for i in range(num_chunks):
        # create empty chunk
        chunk_data = []
        for row in range(CHUNK_ROWS):
            r = [(-1, False)] * CHUNK_COLS
            chunk_data.append(r)

        # load chunk
        with open(f'Chunks/{str(i)}/map.json') as file:
            data = json.load(file)
            for layer in data['layers']:
                for tile in layer['tiles']:
                    x = tile['x']
                    y = tile['y']
                    if x < CHUNK_COLS and y < CHUNK_ROWS and chunk_data[y][x][0] == -1: # IMPORTANT: it seems first layer is closer to the screen so this works
                        chunk_data[y][x] = (int(tile['id']), layer['collider'])

        chunk_spritesheet = pygame.image.load(f'Chunks/{str(i)}/spritesheet.png').convert_alpha()
        tile_img_list = []
        for i in range(NUM_TILE_TYPES):
            tile_img_list.append(get_image(chunk_spritesheet, 8, 16, 16, BLACK, i))

        chunk_datas.append((chunk_data, tile_img_list))

    # def get_cv_image_later(cv_squat_counter, q):
    #     cv_image_rgb, is_cv_jump = cv_squat_counter.get_cv_output()
    #     q.put((cv_image_rgb, is_cv_jump))

    # q = multiprocessing.Queue()
    # processes = []
    # cap = cv2.VideoCapture(0)
    # cv_squat_counter = SquatCounter(cap)

    # Debugging
    # for row in chunk_data:
    #     nums = []
    #     for num in row:
    #         nums.append(num[0])
    #     print(nums)

    # cv_process_started = False

    frame_queue = multiprocessing.Queue()
    squat_queue = multiprocessing.Queue()

    squat_process = multiprocessing.Process(target=squat_detector, args=(frame_queue, squat_queue))
    squat_process.start()

    quit = False
    while not quit:
        #reset varibles
        moving_left = False
        moving_right = False
        world = [] # list of chunks
        screen_scroll = 0
        scroll_offset = 0

        player = Runner('player', 10 * TILE_SIZE, 11 * TILE_SIZE, 3, 10)

        chunk0 = Chunk(0)
        data, tile_img_list = chunk_datas[0]
        chunk0.process_data(data, tile_img_list)
        world.append(chunk0) # always start with the first chunk, maybe add a starter chunk later

        last_chunk_index = 0
        def add_chunks(num, last_chunk_index):
            index = last_chunk_index
            for _ in range(num): # higher number more chunks
                index += 1
                chunk = Chunk(index)
                data, tile_img_list = random.choice(chunk_datas)
                chunk.process_data(data, tile_img_list)
                world.append(chunk) # randomely populate
            return index
            
        last_chunk_index = add_chunks(1, last_chunk_index)

        num_backgrounds = 2 # this is not number of layers, but number of backgrounds side by side

        cam_surface = None

        is_game_loop_running = True
        while is_game_loop_running:

            clock.tick(FPS) # I think it waits to make sure there is a max of 60 frames per second

            if game_start == False:
                screen.fill(BLACK)
                screen.blit(main_menu_img, main_menu_rect)
            else:
                # update background
                if -scroll_offset > bg_images[0].get_width() * (num_backgrounds - 2):
                    num_backgrounds += 1
                draw_bg(num_backgrounds) # used to clear the screen


                if player.rect.left > world[-1].x:
                    last_chunk_index = add_chunks(1, last_chunk_index)

                # draw chunks
                for chunk in world:
                    chunk.draw()

                player.update()
                player.draw()

                draw_score()

                # if q.empty() and not cv_process_started:
                #     cv_process_started = True
                #     process = multiprocessing.Process(target=get_cv_image_later, args=(cv_squat_counter, q,))
                #     process.start()
                #     processes.append(process)
                # elif not q.empty() and cv_process_started:

                #     cv_process_started = False
                #     cv_image_rgb, is_cv_jump = q.get()
                #     cv_image_rgb = np.rot90(cv_image_rgb)
                #     cv_image = pygame.surfarray.make_surface(cv_image_rgb).convert_alpha()
                #     cv_image = pygame.transform.flip(cv_image, True, False)
                #     screen.blit(cv_image, (0,0))

                # Get latest camera frame (non-blocking)
                while not frame_queue.empty():
                    frame = frame_queue.get()
                    frame = np.rot90(frame)  # Rotate for correct orientation
                    cam_surface = pygame.surfarray.make_surface(frame)

                if cam_surface:
                    screen.blit(cam_surface, (0, 0))  # Show camera feed

                # Check squat detection (non-blocking)
                if not squat_queue.empty():
                    squat_detected = squat_queue.get()
                    if squat_detected:
                        player.jump = True
                
                #hello
                #hello
                #hello
                
                if player.alive:
                    #update player actions
                    if player.in_air:
                        player.update_action(2 + player.jump_type) # jump animation
                    elif moving_left or moving_right:
                        player.update_action(1) # run animation
                    else:
                        player.update_action(0) # idle
                else:
                    screen.blit(restart_menu_img, restart_menu_rect)


                screen_scroll = player.move(moving_left, moving_right)
                scroll_offset += screen_scroll
                score = -scroll_offset
                high_score = score if score > high_score else high_score

            for event in pygame.event.get():
                # quit game
                if event.type == pygame.QUIT:
                    is_game_loop_running = False
                    quit = True
                
                # keyboard press
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        moving_left = True
                    if event.key == pygame.K_d:
                        moving_right = True
                    if event.key == pygame.K_w and player.alive:
                        player.jump = True
                    if event.key == pygame.K_ESCAPE:
                        is_game_loop_running = False
                        quit = True
                    if event.key == pygame.K_SPACE and game_start == False:
                        game_start = True
                    if event.key == pygame.K_r and player.alive == False:
                        is_game_loop_running = False
                    if event.key == pygame.K_m and player.alive == False:
                        is_game_loop_running = False
                        game_start = False
                    # return to main menu or restart
    
                # keyboard unpress
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_a:
                        moving_left = False
                    if event.key == pygame.K_d:
                        moving_right = False
            pygame.display.update() # update the screen


    # save highscore

    with open('score.txt', 'w') as file:
        file.write(str(high_score))

    # for process in processes:
    #     process.join()
    # cv_squat_counter.close()
    squat_process.terminate()
    squat_process.join()
    pygame.quit()