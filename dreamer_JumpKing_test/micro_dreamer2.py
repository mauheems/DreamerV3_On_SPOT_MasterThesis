import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
import pygame
import random
import time
import os

# ==========================================
# CONFIGURATION
# ==========================================
# "teach"  = You play -> AI Trains -> AI Saves to disk
# "deploy" = AI Loads from disk -> AI Plays
MODE = "train" 

HUMAN_STEPS = 600    # How long you play
TRAIN_EPOCHS = 1000  # How long AI dreams
FILENAME = "offline_dreamer.pth"

# ==========================================
# 1. THE ENVIRONMENT
# ==========================================
class MiniJumpKing(gym.Env):
    def __init__(self):
        self.window_size = 64
        self.action_space = spaces.Discrete(6) 
        self.observation_space = spaces.Box(0, 255, (3, 64, 64), dtype=np.uint8)
        self.world_height = 320 
        self.world_width = 64
        self.player_width = 6
        self.player_height = 6
        
        self.platforms = [pygame.Rect(0, self.world_height - 10, 64, 10)]
        current_y = self.world_height - 10
        while current_y > 40:
            gap = random.randint(22, 28) 
            current_y -= gap
            w = random.randint(24, 34)   
            x = random.randint(0, 64 - w)
            self.platforms.append(pygame.Rect(x, current_y, w, 4))
        self.platforms.append(pygame.Rect(0, 0, 64, 10)) 
        
        pygame.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.world_surface = pygame.Surface((self.world_width, self.world_height))
        self.reset()

    def reset(self):
        self.player_pos = np.array([32.0, self.world_height - 20.0])
        self.player_vel = np.array([0.0, 0.0])
        self.on_ground = False
        self.jump_cooldown = 0 
        return self._get_obs()

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); exit()

        if self.jump_cooldown > 0: self.jump_cooldown -= 1
        self.player_vel[1] += 0.25 

        if action == 1 or action == 4: self.player_vel[0] = -1.5
        elif action == 2 or action == 5: self.player_vel[0] = 1.5
            
        if self.on_ground and self.jump_cooldown == 0:
            if action in [3, 4, 5]: 
                self.player_vel[1] = -4.2 
                self.on_ground = False
                self.jump_cooldown = 8

        self.player_pos[0] += self.player_vel[0]
        self.player_vel[0] *= 0.8 
        if self.player_pos[0] < 0: self.player_pos[0] = 64
        if self.player_pos[0] > 64: self.player_pos[0] = 0
        self.player_pos[1] += self.player_vel[1]
        
        if self.player_pos[1] < 0: self.player_pos[1] = 0; self.player_vel[1] = 0
        if self.player_pos[1] > self.world_height: self.player_pos[1] = self.world_height - 10; self.player_vel[1] = 0; self.on_ground = True

        player_rect = pygame.Rect(self.player_pos[0], self.player_pos[1], self.player_width, self.player_height)
        if self.jump_cooldown == 0:
            self.on_ground = False 
            for plat in self.platforms:
                if player_rect.colliderect(plat):
                    if self.player_vel[1] > 0 and (self.player_pos[1] + self.player_height) < (plat.bottom):
                        self.player_pos[1] = plat.top - self.player_height
                        self.player_vel[1] = 0
                        self.on_ground = True

        reward = (self.world_height - self.player_pos[1]) / float(self.world_height)
        if self.player_pos[1] < 10: reward += 1.0 
        return self._get_obs(), reward, False, False, {}

    def _get_obs(self):
        self.world_surface.fill((0, 0, 0))
        for p in self.platforms: pygame.draw.rect(self.world_surface, (0, 255, 0), p)
        pygame.draw.rect(self.world_surface, (0, 0, 255), 
                         pygame.Rect(self.player_pos[0], self.player_pos[1], self.player_width, self.player_height))
        
        cam_y = self.player_pos[1] - (self.window_size // 2)
        if cam_y < 0: cam_y = 0
        if cam_y > (self.world_height - self.window_size): cam_y = self.world_height - self.window_size
        
        view = self.world_surface.subsurface(0, cam_y, 64, 64).copy()
        self.window.blit(view, (0,0))
        pygame.display.update()
        return np.transpose(np.array(pygame.surfarray.pixels3d(view)), (2, 0, 1))

# ==========================================
# 2. MODELS
# ==========================================
class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 128), nn.ReLU()
        )
        self.rnn = nn.GRUCell(134, 256) 
        self.reward_head = nn.Linear(256, 1)
        self.decoder_linear = nn.Linear(256, 4096)
        self.decoder_net = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid()
        )
    def forward(self, obs, action, hidden):
        embed = self.encoder(obs / 255.0) 
        action_onehot = F.one_hot(action.long(), 6).float()
        rnn_input = torch.cat([embed, action_onehot], dim=1)
        next_hidden = self.rnn(rnn_input, hidden)
        pred_reward = self.reward_head(next_hidden)
        x = self.decoder_linear(next_hidden)
        pred_image = self.decoder_net(x)
        return pred_image, pred_reward, next_hidden

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 6))
        self.critic = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

def save_checkpoint(world_model, agent):
    torch.save({'world_model': world_model.state_dict(), 'agent': agent.state_dict()}, FILENAME)
    print(f">> SUCCESS: Model saved to {FILENAME}")

def load_checkpoint(world_model, agent, device):
    if os.path.exists(FILENAME):
        checkpoint = torch.load(FILENAME, map_location=device)
        world_model.load_state_dict(checkpoint['world_model'])
        agent.load_state_dict(checkpoint['agent'])
        print(f">> SUCCESS: Model loaded from {FILENAME}")
        return True
    else:
        print(f">> ERROR: No file found at {FILENAME}")
        return False

# ==========================================
# 3. MAIN LOOP
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"MicroDreamer started on {device} in {MODE.upper()} mode.")

    env = MiniJumpKing()
    world_model = WorldModel().to(device)
    agent = ActorCritic().to(device)
    opt_world = optim.Adam(world_model.parameters(), lr=1e-3)
    opt_agent = optim.Adam(agent.parameters(), lr=1e-4)
    
    # ==========================================
    # MODE: TEACH (Play -> Train -> Save)
    # ==========================================
    if MODE == "teach":
        buffer = []
        
        # 1. HUMAN COLLECTION
        print(f"\n[PHASE 1] HUMAN TEACHER MODE")
        print("Controls: Arrows to Walk, Space=Jump, Z=JumpL, X=JumpR")
        obs = env.reset()
        hidden = torch.zeros(1, 256).to(device) 
        
        for step in range(HUMAN_STEPS):
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            action = 0 
            if keys[pygame.K_LEFT]: action = 1
            elif keys[pygame.K_RIGHT]: action = 2
            elif keys[pygame.K_SPACE]: action = 3
            elif keys[pygame.K_z]: action = 4
            elif keys[pygame.K_x]: action = 5
            
            next_obs, reward, _, _, _ = env.step(action)
            buffer.append((obs, action, reward, next_obs))
            obs = next_obs
            
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                act_t = torch.tensor([action]).to(device)
                _, _, hidden = world_model(obs_t, act_t, hidden)
            time.sleep(0.05)

        print(f"\n[PHASE 1 COMPLETE] Collected {len(buffer)} examples.")
        
        # 2. OFFLINE TRAINING
        print(f"\n[PHASE 2] OFFLINE TRAINING (DREAMING)")
        pygame.display.set_caption("MicroDreamer - TRAINING (Please Wait)")
        train_hidden = torch.zeros(32, 256).to(device)

        for epoch in range(TRAIN_EPOCHS):
            batch = random.sample(buffer, 32)
            b_obs = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(device)
            b_act = torch.tensor(np.array([x[1] for x in batch])).to(device)
            b_rew = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).unsqueeze(1).to(device)
            b_hid = train_hidden.detach() 

            # Train World Model
            pred_img, pred_rew, next_hidden_batch = world_model(b_obs, b_act, b_hid)
            loss_world = F.mse_loss(pred_img, b_obs / 255.0) + F.mse_loss(pred_rew, b_rew)
            opt_world.zero_grad(); loss_world.backward(); opt_world.step()
            
            # Train Actor
            for p in world_model.parameters(): p.requires_grad = False
            dream_hid = b_hid.detach(); dream_loss = 0
            for _ in range(15): 
                action_logits = agent.actor(dream_hid)
                act_dream = torch.distributions.Categorical(logits=action_logits).sample()
                _, _, dream_hid = world_model(b_obs, act_dream, dream_hid) 
                value = agent.critic(dream_hid)
                dream_loss -= value.mean()
            opt_agent.zero_grad(); dream_loss.backward(); opt_agent.step()
            for p in world_model.parameters(): p.requires_grad = True
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{TRAIN_EPOCHS} | Loss: {loss_world.item():.4f}")

        # 3. SAVE
        save_checkpoint(world_model, agent)
        print("\n[READY] You can now set MODE='deploy' and run again.")

    # ==========================================
    # MODE: DEPLOY (Load -> Play)
    # ==========================================
    elif MODE == "deploy":
        if not load_checkpoint(world_model, agent, device):
            return

        print(f"\n[PHASE 3] AI TAKEOVER")
        pygame.display.set_caption("MicroDreamer - AI PLAYING")
        obs = env.reset()
        hidden = torch.zeros(1, 256).to(device) 
        
        while True:
            with torch.no_grad():
                logits = agent.actor(hidden)
                action = torch.argmax(logits).item() # Greedy (Best) Action

            next_obs, reward, _, _, _ = env.step(action)
            
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                act_t = torch.tensor([action]).to(device)
                _, _, hidden = world_model(obs_t, act_t, hidden)

            obs = next_obs
            time.sleep(0.05)

if __name__ == "__main__":
    main()