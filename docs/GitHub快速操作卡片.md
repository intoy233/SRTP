# 🚀 GitHub快速操作卡片

## 📋 每日必用命令

### 🔄 开始工作前
```bash
# 1. 进入项目目录
cd bridge-vortex-risk-assessment

# 2. 拉取最新代码
git pull origin main
```

### 💾 提交你的工作
```bash
# 1. 查看修改状态
git status

# 2. 添加文件
git add .                    # 添加所有修改
git add 文件名.csv           # 添加特定文件

# 3. 提交修改
git commit -m "描述你的修改"

# 4. 推送到GitHub
git push origin main
```

---

## 📁 常用文件操作

### 添加PDF论文
```bash
# 将PDF放入正确位置
cp "论文.pdf" data/raw/papers/

# 提交
git add data/raw/papers/论文.pdf
git commit -m "添加论文：论文标题"
git push origin main
```

### 添加数据文件
```bash
# 将CSV放入正确位置
cp "数据.csv" data/raw/extracted/

# 提交
git add data/raw/extracted/数据.csv
git commit -m "添加数据：桥梁名称"
git push origin main
```

### 更新个人进度
```bash
# 编辑进度文件
# docs/个人进度/你的姓名_进度.md

# 提交
git add docs/个人进度/你的姓名_进度.md
git commit -m "更新个人进度"
git push origin main
```

---

## ⚡ 快速解决问题

### 问题1：推送失败
```bash
# 先拉取，再推送
git pull origin main
git push origin main
```

### 问题2：忘记拉取就开始工作
```bash
# 暂存工作
git stash

# 拉取最新代码
git pull origin main

# 恢复工作
git stash pop
```

### 问题3：撤销最后一次提交
```bash
# 撤销提交，保留修改
git reset --soft HEAD~1
```

---

## 📝 提交信息模板

### 好的提交信息示例
```
✅ 添加润扬大桥涡振数据 - 包含3个测点
✅ 修复数据模板中的单位错误
✅ 更新个人进度 - 完成第一周目标
✅ 上传桥梁涡振分析论文 - 张三2023
✅ 完善数据收集指南文档
```

### 避免的提交信息
```
❌ 更新
❌ 修改
❌ 临时保存
❌ test
❌ 123
```

---

## 🗂️ 文件命名规范

### PDF论文
```
格式：论文标题_第一作者_年份.pdf
示例：大跨度悬索桥涡振响应分析_张三_2023.pdf
```

### CSV数据
```
格式：桥梁名称_数据类型_日期.csv
示例：润扬大桥_涡振数据_20241201.csv
```

### 个人进度
```
格式：姓名_进度.md
示例：张三_进度.md
```

---

## 🎯 每日工作流程

### ⏰ 开始工作 (5分钟)
1. 打开命令行
2. `cd bridge-vortex-risk-assessment`
3. `git pull origin main`
4. 开始你的工作

### 💼 工作中 (随时)
- 完成一个小任务就提交一次
- 写清楚提交信息
- 遇到问题及时沟通

### 🏁 结束工作 (5分钟)
1. `git status` 检查状态
2. `git add .` 添加所有修改
3. `git commit -m "今天的工作总结"`
4. `git push origin main`

---

## 📞 紧急联系

### 遇到Git问题
1. 📖 查看《GitHub协作指南.md》
2. 🔍 搜索：`git 错误信息 解决方案`
3. 💬 微信群求助
4. 📱 联系项目负责人

### 常见错误代码
- `error: failed to push` → 先pull再push
- `fatal: not a git repository` → 检查目录
- `merge conflict` → 手动解决冲突

---

## 🏆 协作小贴士

### ✅ 好习惯
- 每天至少提交一次
- 工作前先pull
- 写清楚提交信息
- 及时沟通问题

### ❌ 避免
- 长时间不提交
- 提交信息不清楚
- 直接修改别人的文件
- 上传大文件(>100MB)

---

## 🎮 Git命令速查

| 命令 | 作用 | 使用场景 |
|------|------|----------|
| `git status` | 查看状态 | 随时检查 |
| `git add .` | 添加所有修改 | 提交前 |
| `git commit -m "信息"` | 提交修改 | 完成工作 |
| `git push origin main` | 推送到GitHub | 分享工作 |
| `git pull origin main` | 拉取最新代码 | 开始工作前 |
| `git log --oneline` | 查看提交历史 | 了解进度 |
| `git stash` | 暂存工作 | 紧急切换 |
| `git stash pop` | 恢复工作 | 继续之前工作 |

---

**💡 记住：遇到问题不要慌，先查文档，再求助！**

**🎯 目标：让每个人都能轻松使用GitHub协作！**