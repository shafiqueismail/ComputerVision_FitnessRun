import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen settings
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 450
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Parallax Background")

# Load images
backgrounds = [
    pygame.image.load("Images/Backgrounds_Sky.png"),             
    pygame.image.load("Images/Backgrounds_BuildingsFar.png"),
    pygame.image.load("Images/Backgrounds_BuildingsBack.png"),
    pygame.image.load("Images/Backgrounds_BuildingsMid.png"),
    pygame.image.load("Images/Backgrounds_BuildingsClose.png")
]

# Scale images to fit screen height
backgrounds = [pygame.transform.scale(bg, (bg.get_width(), SCREEN_HEIGHT)) for bg in backgrounds]

# Parallax speeds (lower values move slower, higher move faster)
speeds = [0.2, 0.5, 0.8, 1.2, 1.5]  # Removed moon speed

# Initial x positions of the backgrounds
positions = [0] * len(backgrounds)

# Game loop
clock = pygame.time.Clock()
running = True
while running:
    screen.fill((0, 0, 0))
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Update positions and draw images
    for i in range(len(backgrounds)):
        positions[i] += speeds[i]
        if positions[i] >= SCREEN_WIDTH:
            positions[i] = 0
        
        # Draw images twice to create an infinite scrolling effect
        screen.blit(backgrounds[i], (-positions[i], 0))
        screen.blit(backgrounds[i], (SCREEN_WIDTH - positions[i], 0))
    
    pygame.display.update()
    clock.tick(60)  # Maintain 60 FPS

pygame.quit()
sys.exit()
