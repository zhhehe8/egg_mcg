"""
贪吃蛇游戏
使用pygame库实现经典的贪吃蛇游戏

控制方式：
- 方向键或WASD控制蛇的移动
- ESC键退出游戏
- 空格键重新开始游戏（游戏结束后）

游戏规则：
- 吃到食物后蛇身变长，得分增加
- 撞到墙壁或自己身体游戏结束
- 速度会随着得分增加而提升
"""

import pygame
import random
import sys

# 初始化pygame
pygame.init()

# 颜色定义 (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

# 游戏设置
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CELL_SIZE = 20
CELL_NUMBER_X = WINDOW_WIDTH // CELL_SIZE
CELL_NUMBER_Y = WINDOW_HEIGHT // CELL_SIZE

# 方向定义
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    """贪吃蛇类"""
    
    def __init__(self):
        """初始化蛇"""
        self.reset()
    
    def reset(self):
        """重置蛇的状态"""
        # 蛇身体，每个元素是(x, y)坐标
        self.body = [(CELL_NUMBER_X // 2, CELL_NUMBER_Y // 2)]
        self.direction = RIGHT
        self.new_block = False  # 是否需要增长
    
    def draw(self, screen):
        """绘制蛇"""
        for segment in self.body:
            x = segment[0] * CELL_SIZE
            y = segment[1] * CELL_SIZE
            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GREEN, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)  # 边框
    
    def move(self):
        """移动蛇"""
        # 获取头部位置
        head_x, head_y = self.body[0]
        
        # 计算新的头部位置
        new_head = (head_x + self.direction[0], head_y + self.direction[1])
        
        # 添加新头部
        self.body.insert(0, new_head)
        
        # 如果没有吃到食物，移除尾部
        if not self.new_block:
            self.body.pop()
        else:
            self.new_block = False
    
    def change_direction(self, new_direction):
        """改变蛇的方向"""
        # 防止蛇反向移动
        if (new_direction[0] * -1, new_direction[1] * -1) != self.direction:
            self.direction = new_direction
    
    def check_collision(self):
        """检查碰撞"""
        head_x, head_y = self.body[0]
        
        # 检查是否撞墙
        if (head_x < 0 or head_x >= CELL_NUMBER_X or 
            head_y < 0 or head_y >= CELL_NUMBER_Y):
            return True
        
        # 检查是否撞到自己
        if self.body[0] in self.body[1:]:
            return True
        
        return False
    
    def grow(self):
        """蛇增长"""
        self.new_block = True

class Food:
    """食物类"""
    
    def __init__(self):
        """初始化食物"""
        self.randomize()
    
    def randomize(self):
        """随机生成食物位置"""
        self.x = random.randint(0, CELL_NUMBER_X - 1)
        self.y = random.randint(0, CELL_NUMBER_Y - 1)
        self.pos = (self.x, self.y)
    
    def draw(self, screen):
        """绘制食物"""
        x = self.x * CELL_SIZE
        y = self.y * CELL_SIZE
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, RED, rect)
        pygame.draw.rect(screen, BLACK, rect, 2)  # 边框

class Game:
    """游戏主类"""
    
    def __init__(self):
        """初始化游戏"""
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("贪吃蛇游戏")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.font_large = pygame.font.Font(None, 72)
        
        self.reset_game()
    
    def reset_game(self):
        """重置游戏"""
        self.snake = Snake()
        self.food = Food()
        self.score = 0
        self.game_over = False
        self.speed = 5  # 初始速度
        
        # 确保食物不会生成在蛇身上
        while self.food.pos in self.snake.body:
            self.food.randomize()
    
    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                
                elif event.key == pygame.K_SPACE and self.game_over:
                    self.reset_game()
                
                elif not self.game_over:
                    # 方向键控制
                    if event.key == pygame.K_UP or event.key == pygame.K_w:
                        self.snake.change_direction(UP)
                    elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                        self.snake.change_direction(DOWN)
                    elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        self.snake.change_direction(LEFT)
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        self.snake.change_direction(RIGHT)
        
        return True
    
    def update(self):
        """更新游戏状态"""
        if not self.game_over:
            # 移动蛇
            self.snake.move()
            
            # 检查是否吃到食物
            if self.snake.body[0] == self.food.pos:
                self.score += 10
                self.snake.grow()
                self.food.randomize()
                
                # 确保新食物不会生成在蛇身上
                while self.food.pos in self.snake.body:
                    self.food.randomize()
                
                # 增加速度
                self.speed = min(15, 5 + self.score // 50)
            
            # 检查碰撞
            if self.snake.check_collision():
                self.game_over = True
    
    def draw(self):
        """绘制游戏画面"""
        # 清屏
        self.screen.fill(BLACK)
        
        if not self.game_over:
            # 绘制蛇和食物
            self.snake.draw(self.screen)
            self.food.draw(self.screen)
            
            # 绘制分数
            score_text = self.font.render(f"得分: {self.score}", True, WHITE)
            self.screen.blit(score_text, (10, 10))
            
            # 绘制速度
            speed_text = self.font.render(f"速度: {self.speed}", True, WHITE)
            self.screen.blit(speed_text, (10, 50))
        
        else:
            # 游戏结束画面
            game_over_text = self.font_large.render("游戏结束!", True, RED)
            final_score_text = self.font.render(f"最终得分: {self.score}", True, WHITE)
            restart_text = self.font.render("按空格键重新开始，ESC键退出", True, YELLOW)
            
            # 居中显示文本
            game_over_rect = game_over_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 - 50))
            score_rect = final_score_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2))
            restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 50))
            
            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(final_score_text, score_rect)
            self.screen.blit(restart_text, restart_rect)
        
        # 更新显示
        pygame.display.flip()
    
    def run(self):
        """运行游戏主循环"""
        running = True
        
        print("贪吃蛇游戏开始!")
        print("控制方式: 方向键或WASD")
        print("ESC键退出，空格键重新开始")
        
        while running:
            # 处理事件
            running = self.handle_events()
            
            # 更新游戏状态
            self.update()
            
            # 绘制画面
            self.draw()
            
            # 控制帧率
            self.clock.tick(self.speed)
        
        pygame.quit()
        sys.exit()

def main():
    """主函数"""
    try:
        game = Game()
        game.run()
    except Exception as e:
        print(f"游戏运行出错: {e}")
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
