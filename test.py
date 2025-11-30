import gymnasium as gym
import numpy as np
import cv2
import time

# ====================== 完全对齐训练的核心配置 ======================
ENV_NAME = "ALE/MontezumaRevenge-v5"
RENDER_SIZE = (600, 800)
BASE_FPS = 60  # Atari原生基础帧率
ACTION_FRAME_SKIP = 4  # 关键：每个动作持续执行4帧（与训练一致）
RENDER_INTERVAL = int(1000 / BASE_FPS)  # 单帧渲染间隔（匹配原生帧率）

def resize_frame(frame, target_size):
    """仅调整帧大小，无其他修改"""
    h, w = frame.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    offset_x = (target_size[0] - new_w) // 2
    offset_y = (target_size[1] - new_h) // 2
    canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
    return canvas

def draw_raw_info(frame, action_num, action_meaning, step, reward, total_reward, all_action_meanings, frame_skip):
    """绘制信息，新增帧跳步提示"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    white = (255, 255, 255)
    black = (0, 0, 0)
    thickness = 2
    
    # 生成字母键映射说明
    letter_mapping_hint = "字母键映射：a=10, b=11, c=12, ..., z=35"
    # 仅显示游戏原生信息（新增帧跳步提示）
    info = [
        f"【原生动作信息】",
        f"动作数字: {action_num} | 持续执行: {frame_skip}帧（共{ACTION_FRAME_SKIP}帧）",
        f"动作含义(游戏内置): {action_meaning}",
        f"当前步数(动作数): {step} | 累计奖励: {total_reward:.2f}",
        "",
        f"【核心逻辑】：每个动作持续执行{ACTION_FRAME_SKIP}帧（与训练一致）",
        f"【操作说明】",
        f"按 0-9 键: 执行动作0-9 | {letter_mapping_hint}",
        f"按 S 键: 输入动作序列批量执行",
        f"按 R 键: 重置游戏",
        f"按 ESC 键: 退出",
        f"按 ↑↓←→ 键: 快速测试上下左右（仅参考）"
    ]
    
    y = 30
    for text in info:
        (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, (10, y-h-5), (10+w+10, y+5), black, -1)
        cv2.putText(frame, text, (15, y), font, font_scale, white, thickness)
        y += 30
    
    # 绘制所有原生动作含义列表（便于对照）
    y += 10
    cv2.putText(frame, "【所有原生动作含义】", (15, y), font, 0.7, (0, 255, 0), thickness)
    y += 35
    for idx, meaning in enumerate(all_action_meanings):
        text = f"{idx}: {meaning}"
        (w, h), _ = cv2.getTextSize(text, font, 0.5, 1)
        cv2.rectangle(frame, (10, y-h-3), (10+w+10, y+3), black, -1)
        cv2.putText(frame, text, (15, y), font, 0.5, white, 1)
        y += 20
        # 每行显示4个，换行
        if (idx + 1) % 4 == 0:
            y += 5
    return frame

def execute_action_for_4_frames(env, action_num, all_action_meanings):
    """核心函数：单个动作执行4帧（与训练一致），返回累计奖励和结束状态"""
    total_reward = 0.0
    terminated = False
    truncated = False
    
    # 严格校验动作数字
    if action_num < 0 or action_num >= len(all_action_meanings):
        print(f"❌ 动作数字{action_num}超出范围（游戏原生动作数：{len(all_action_meanings)}）")
        return 0.0, False, False
    
    action_meaning = all_action_meanings[action_num]
    print(f"\n▶️  执行动作 {action_num}（{action_meaning}），持续{ACTION_FRAME_SKIP}帧")
    
    # 执行4帧相同动作（训练时的frame skip逻辑）
    for frame_idx in range(ACTION_FRAME_SKIP):
        try:
            obs, reward, terminated, truncated, info = env.step(action_num)
            total_reward += reward
            
            # 每帧都渲染（保持视觉流畅）
            frame = env.render()
            frame = resize_frame(frame, RENDER_SIZE)
            frame = draw_raw_info(frame, action_num, action_meaning, 0, reward, total_reward, all_action_meanings, frame_idx+1)
            cv2.imshow("RAW Atari Action Tester", frame)
            
            # 控制单帧帧率，匹配原生游戏速度
            cv2.waitKey(RENDER_INTERVAL)
            
            # 如果游戏结束，提前终止
            if terminated or truncated:
                print(f"⚠️  第{frame_idx+1}帧时游戏结束，终止动作执行")
                break
        except Exception as e:
            print(f"❌ 执行动作{action_num}第{frame_idx+1}帧失败：{e}")
            break
    
    print(f"✅ 动作{action_num}执行完成 | 4帧累计奖励: {total_reward:.2f}")
    return total_reward, terminated, truncated

def key_to_action_num(key):
    """将键盘按键转换为动作数字：0-9→0-9，a-z→10-35"""
    # 数字键 0-9
    if 48 <= key <= 57:
        return key - 48
    # 小写字母 a-z
    elif 97 <= key <= 122:
        return 10 + (key - 97)
    # 大写字母 A-Z（兼容）
    elif 65 <= key <= 90:
        return 10 + (key - 65)
    # 无效键
    else:
        return -1

def manual_raw_test(env, all_action_meanings):
    """纯手动测试：单个动作执行4帧（完全对齐训练）"""
    print("\n=====================================")
    print("【纯原生动作测试模式（匹配训练逻辑）】")
    print("=====================================")
    print(f"游戏: {ENV_NAME}")
    print(f"原生动作总数: {len(all_action_meanings)}")
    print("📌 核心逻辑（与训练一致）：")
    print(f"   单个动作按下后，会持续执行{ACTION_FRAME_SKIP}帧")
    print("📌 按键映射规则：")
    print("   数字键 0-9 → 动作 0-9")
    print("   字母键 a-z → 动作 10-35（a=10, b=11, ..., z=35）")
    print("   字母键 A-Z → 同小写（兼容大写）")
    print("原生动作列表（数字 ↔ 游戏内置含义）：")
    for idx, meaning in enumerate(all_action_meanings):
        print(f"  {idx}: {meaning}")
    print("\n操作规则：")
    print(f"1. 按 0-9/a-z 键，执行对应数字的原生动作（持续{ACTION_FRAME_SKIP}帧）")
    print(f"2. 按 S 键：输入动作序列批量执行")
    print(f"3. 按 R 键：重置游戏")
    print(f"4. 按 ESC 键：退出")
    print("=====================================\n")
    
    # 初始化所有变量
    obs, info = env.reset()
    total_reward = 0.0
    action_step = 0  # 动作步数（不是帧数）
    current_action_num = 0
    current_action_meaning = all_action_meanings[0]
    
    cv2.namedWindow("RAW Atari Action Tester", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("RAW Atari Action Tester", RENDER_SIZE[0], RENDER_SIZE[1])
    
    while True:
        # 初始渲染（无动作时显示）
        frame = env.render()
        frame = resize_frame(frame, RENDER_SIZE)
        frame = draw_raw_info(frame, current_action_num, current_action_meaning, 
                              action_step, 0, total_reward, all_action_meanings, 0)
        cv2.imshow("RAW Atari Action Tester", frame)
        
        # 等待按键输入（无超时，确保按一次键执行一次动作）
        key = cv2.waitKey(0) & 0xFF
        
        # 退出
        if key == 27:  # ESC
            break
        
        # 重置游戏
        if key == ord('r') or key == ord('R'):
            obs, info = env.reset()
            total_reward = 0.0
            action_step = 0
            current_action_num = 0
            current_action_meaning = all_action_meanings[0]
            print("\n🔄 游戏已重置")
            continue
        
        # 进入序列执行模式
        if key == ord('s') or key == ord('S'):
            cv2.destroyAllWindows()
            return "sequence"
        
        # 处理方向键（仅参考，执行对应动作的4帧）
        action_num = -1
        if key == 2490368:  # 上
            action_num = [idx for idx, m in enumerate(all_action_meanings) if "UP" in m][0] if any("UP" in m for m in all_action_meanings) else 0
        elif key == 2621440:  # 下
            action_num = [idx for idx, m in enumerate(all_action_meanings) if "DOWN" in m][0] if any("DOWN" in m for m in all_action_meanings) else 0
        elif key == 2424832:  # 左
            action_num = [idx for idx, m in enumerate(all_action_meanings) if "LEFT" in m][0] if any("LEFT" in m for m in all_action_meanings) else 0
        elif key == 2555904:  # 右
            action_num = [idx for idx, m in enumerate(all_action_meanings) if "RIGHT" in m][0] if any("RIGHT" in m for m in all_action_meanings) else 0
        else:
            # 处理数字键+字母键（核心）
            action_num = key_to_action_num(key)
        
        # 执行动作（4帧）
        if action_num != -1:
            if action_num < len(all_action_meanings):
                current_action_num = action_num
                current_action_meaning = all_action_meanings[action_num]
                
                # 执行4帧动作，获取累计奖励和结束状态
                reward_4frames, terminated, truncated = execute_action_for_4_frames(
                    env, action_num, all_action_meanings
                )
                
                # 更新统计（按动作步数计数，不是帧数）
                total_reward += reward_4frames
                action_step += 1
                
                # 游戏结束自动重置
                if terminated or truncated:
                    print(f"\n🎮 游戏结束 | 总动作步数: {action_step} | 总奖励: {total_reward:.2f}")
                    obs, info = env.reset()
                    total_reward = 0.0
                    action_step = 0
                    current_action_num = 0
                    current_action_meaning = all_action_meanings[0]
            else:
                print(f"❌ 动作数字{action_num}超出游戏原生动作范围（最大：{len(all_action_meanings)-1}）")
    
    cv2.destroyAllWindows()
    return "exit"

def sequence_raw_test(env, all_action_meanings):
    """序列测试：每个动作执行4帧（匹配训练）"""
    print("\n=====================================")
    print("【原生动作序列执行模式（匹配训练）】")
    print("=====================================")
    print(f"游戏原生动作总数: {len(all_action_meanings)}")
    print(f"核心逻辑：序列中每个动作都会持续执行{ACTION_FRAME_SKIP}帧")
    
    while True:
        seq_input = input("请输入原生动作数字序列: ").strip()
        
        # 返回手动模式
        if seq_input.lower() == 'q':
            return "manual"
        
        # 退出
        if seq_input.lower() == 'esc':
            return "exit"
        
        # 解析序列
        try:
            action_seq = [int(num.strip()) for num in seq_input.split(',')]
            # 严格校验
            valid = True
            for num in action_seq:
                if num < 0 or num >= len(all_action_meanings):
                    print(f"❌ 无效数字：{num}（游戏原生动作范围：0-{len(all_action_meanings)-1}）")
                    valid = False
                    break
            if not valid:
                continue
        except ValueError:
            print("❌ 输入格式错误！请输入纯数字序列（用逗号分隔），如：0,3,10")
            continue
        
        # 执行序列（每个动作4帧）
        print(f"\n▶️  开始执行动作序列：{action_seq} | 每个动作执行{ACTION_FRAME_SKIP}帧")
        obs, info = env.reset()
        total_reward = 0.0
        action_step = 0
        terminated = False
        truncated = False
        
        cv2.namedWindow("RAW Atari Action Tester", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("RAW Atari Action Tester", RENDER_SIZE[0], RENDER_SIZE[1])
        
        for action_num in action_seq:
            if terminated or truncated:
                break
            
            # 单个动作执行4帧
            reward_4frames, terminated, truncated = execute_action_for_4_frames(
                env, action_num, all_action_meanings
            )
            
            # 更新统计
            total_reward += reward_4frames
            action_step += 1
            
            # 检查是否中途退出
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                print("\n⏹️  序列执行被中断")
                terminated = True
                break
        
        cv2.destroyAllWindows()
        print(f"\n✅ 序列执行完成 | 总动作步数: {action_step} | 总奖励: {total_reward:.2f}")

def main():
    # 初始化环境（完全对齐训练配置）
    try:
        # 禁用默认frame skip，手动控制4帧执行（避免重复跳帧）
        env = gym.make(ENV_NAME, render_mode="rgb_array", frameskip=1)
        # 手动设置render_fps，消除警告
        env.metadata['render_fps'] = BASE_FPS
    except Exception as e:
        print(f"❌ 加载游戏失败：{e}")
        print("💡 请安装依赖：pip install gymnasium[atari] ale-py==0.8.1")
        return
    
    # 获取原生动作含义
    all_action_meanings = env.unwrapped.get_action_meanings()
    
    print("=====================================")
    print("【Atari 动作测试工具（匹配训练逻辑）】")
    print("=====================================")
    print(f"游戏名称: {ENV_NAME}")
    print(f"原生动作数量: {env.action_space.n}")
    print(f"核心配置（与训练一致）：")
    print(f"   单个动作持续执行 {ACTION_FRAME_SKIP} 帧")
    print(f"   游戏基础帧率: {BASE_FPS} FPS")
    print(f"📌 按键映射规则：")
    print(f"   数字键 0-9 → 动作 0-9")
    print(f"   字母键 a-z → 动作 10-35（a=10, b=11, ..., z=35）")
    print(f"原生动作含义（游戏内置）:")
    for idx, meaning in enumerate(all_action_meanings):
        print(f"  {idx}: {meaning}")
    print("=====================================\n")
    
    # 主循环
    mode = "manual"
    while True:
        if mode == "manual":
            mode = manual_raw_test(env, all_action_meanings)
        elif mode == "sequence":
            mode = sequence_raw_test(env, all_action_meanings)
        elif mode == "exit":
            break
    
    # 清理
    env.close()
    print("\n👋 测试结束")

if __name__ == "__main__":
    # 检查依赖
    try:
        import ale_py
    except ImportError:
        print("⚠️  缺少ALE依赖，请执行：")
        print("pip install gymnasium[atari] ale-py==0.8.1")
        exit(1)
    
    main()
