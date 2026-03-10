# 🚀 Protocol-algorithm v2.0 - GitHub 推送与分支设置指南

---

## ✅ 本地状态

**Git 提交已完成:**
- **Commit:** 7cb2254
- **文件:** 60 个
- **代码:** ~5,500 行
- **状态:** ✅ 准备就绪

---

## 📋 推送步骤

### 步骤 1: 进入项目目录

```bash
cd /home/node/.openclaw/workspace-dev-planner/protocol-algorithm-v2
```

### 步骤 2: 检查远程仓库

```bash
git remote -v
```

**如果没有远程仓库，添加:**

```bash
git remote add origin https://github.com/xfengyin/Protocol-algorithm.git
```

### 步骤 3: 推送到 GitHub

```bash
# 推送到 main 分支
git push -u origin main

# 如果遇到网络问题，使用 token 方式:
git push https://<your-username>:<your-token>@github.com/xfengyin/Protocol-algorithm.git main
```

**GitHub Token 获取:**
1. 访问 https://github.com/settings/tokens
2. Generate new token (classic)
3. 勾选 `repo` 权限
4. 复制 token (只显示一次)

### 步骤 4: 设置默认分支

推送成功后，访问 GitHub 仓库设置：

1. 打开 https://github.com/xfengyin/Protocol-algorithm/settings
2. 找到 "Default branch"
3. 点击 "Switch to main"
4. 确认切换

**或使用 GitHub CLI:**

```bash
# 安装 gh
gh auth login

# 设置默认分支
gh api -X PATCH repos/xfengyin/Protocol-algorithm -f default_branch=main
```

---

## 🔀 保留旧版本

### 当前远程分支状态

```bash
# 查看远程分支
git fetch origin
git branch -r
```

**应该显示:**
```
  origin/master
  origin/feature/ui-enhancements
  origin/feature/uv-packaging
  origin/main (新推送)
```

### 保留旧分支

**不需要删除任何旧分支！** 它们会自动保留：
- ✅ `master` - 旧版本 (保留)
- ✅ `feature/ui-enhancements` - 功能分支 (保留)
- ✅ `feature/uv-packaging` - 功能分支 (保留)

---

## 📊 分支结构

```
Protocol-algorithm
├── main (新默认分支) ✅ v2.0 重构版
│   ├── Rust 核心
│   ├── Python 绑定
│   ├── Web 前端
│   └── 完整文档
│
├── master (旧版本，保留) ⏸️ v1.0
│   └── 原始实现
│
└── feature/* (功能分支，保留) 🔧
    ├── ui-enhancements
    └── uv-packaging
```

---

## 🎯 推送后验证

### 1. 检查 GitHub 仓库

访问 https://github.com/xfengyin/Protocol-algorithm

**确认:**
- ✅ 默认分支显示为 `main`
- ✅ 最新提交是 "feat: v2.0 完整重构 - 混合架构实现"
- ✅ 文件列表包含 60 个文件
- ✅ `master` 分支仍然存在

### 2. 检查文件结构

确认以下目录存在:
- ✅ `core/` - Rust 核心
- ✅ `python/` - Python 绑定
- ✅ `web/` - Web 前端
- ✅ `viz/` - 可视化脚本
- ✅ `docs/` - 文档
- ✅ `.github/workflows/` - CI/CD
- ✅ `Dockerfile` - Docker 配置
- ✅ `docker-compose.yml` - Docker 编排

### 3. 检查 README

确认 README.md 显示在仓库首页，包含:
- ✅ 项目介绍
- ✅ 快速开始
- ✅ 使用示例
- ✅ 性能对比
- ✅ 架构图

---

## 🔧 故障排除

### 问题 1: 认证失败

**错误:**
```
fatal: Authentication failed for 'https://github.com/xfengyin/Protocol-algorithm.git/'
```

**解决:**
```bash
# 使用 token 代替密码
git push https://xfengyin:<TOKEN>@github.com/xfengyin/Protocol-algorithm.git main

# 或使用 SSH (如果配置了)
git remote set-url origin git@github.com:xfengyin/Protocol-algorithm.git
git push -u origin main
```

### 问题 2: 网络连接超时

**错误:**
```
fatal: unable to access 'https://github.com/xfengyin/Protocol-algorithm.git/': 
Failed to connect to github.com port 443 after 134255 ms
```

**解决:**
1. 检查网络连接
2. 尝试使用代理
3. 稍后重试
4. 使用 GitHub Desktop 推送

### 问题 3: 分支已存在

**错误:**
```
error: src refspec main does not match any
```

**解决:**
```bash
# 重命名当前分支为 main
git branch -M main

# 再次推送
git push -u origin main
```

### 问题 4: 需要强制推送

**错误:**
```
! [rejected] main -> main (fetch first)
```

**解决:**
```bash
# 先拉取远程变更
git pull origin main --rebase

# 或强制推送 (谨慎使用)
git push -u origin main --force
```

---

## 📝 完整命令清单

**一键执行 (复制粘贴):**

```bash
# 1. 进入目录
cd /home/node/.openclaw/workspace-dev-planner/protocol-algorithm-v2

# 2. 确认提交
git log --oneline -1

# 3. 添加远程仓库 (如果没有)
git remote add origin https://github.com/xfengyin/Protocol-algorithm.git

# 4. 推送
git push -u origin main

# 5. 验证
git branch -r
```

---

## 🎊 推送后操作清单

### GitHub 设置

- [ ] 访问仓库 Settings
- [ ] 将默认分支改为 `main`
- [ ] 确认 `master` 分支仍然存在
- [ ] 启用 GitHub Actions (CI/CD)

### GitHub Pages (可选)

- [ ] Settings → Pages
- [ ] Source: Deploy from branch
- [ ] Branch: main
- [ ] Folder: /
- [ ] Save

### GitHub Actions (可选)

- [ ] Actions → 同意启用 workflows
- [ ] 等待 CI 运行完成
- [ ] 查看测试结果

### 创建 Release (可选)

- [ ] Releases → Create a new release
- [ ] Tag: v2.0.0-alpha
- [ ] Target: main
- [ ] 填写发布说明
- [ ] Publish

---

## 📊 项目统计

| 类别 | 数量 | 备注 |
|------|------|------|
| **总文件** | 60 | 包含配置、测试、文档 |
| **代码行数** | ~5,500 | 不含空白行 |
| **Rust 文件** | 18 | ~1,200 行 |
| **TypeScript** | 10 | ~600 行 |
| **Python** | 8 | ~500 行 |
| **配置** | 12 | TOML/JSON/YAML |
| **文档** | 9 | ~1,500 行 |
| **CI/CD** | 1 | GitHub Actions |
| **Docker** | 2 | Dockerfile + Compose |

---

## 🔗 相关链接

- **仓库:** https://github.com/xfengyin/Protocol-algorithm
- **本地路径:** `/home/node/.openclaw/workspace-dev-planner/protocol-algorithm-v2/`
- **提交哈希:** 7cb2254
- **版本:** v2.0.0-alpha

---

## ✅ 完成确认

推送成功后，仓库应该显示：

**默认分支:** `main` ✅  
**最新提交:** "feat: v2.0 完整重构 - 混合架构实现" ✅  
**旧分支保留:** `master`, `feature/*` ✅  
**文件数量:** 60 个 ✅  

---

**准备就绪，请执行推送命令！** 🚀

_项目完成时间：2026-03-10_  
_版本：v2.0.0-alpha_  
_重构方案：D (混合架构)_
