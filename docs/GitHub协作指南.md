# 🚀 桥梁涡振风险评估项目 - GitHub协作指南

## 📋 目录
1. [项目负责人操作指南](#项目负责人操作指南)
2. [组员加入项目指南](#组员加入项目指南)
3. [日常协作流程](#日常协作流程)
4. [文件管理规范](#文件管理规范)
5. [常见问题解决](#常见问题解决)
6. [协作最佳实践](#协作最佳实践)

---

## 🎯 项目负责人操作指南

### 第一步：创建GitHub仓库

#### 1.1 登录GitHub
- 访问 [github.com](https://github.com)
- 登录你的GitHub账号（如果没有账号，先注册一个）

#### 1.2 创建新仓库
1. 点击右上角的 "+" 号，选择 "New repository"
2. 填写仓库信息：
   - **Repository name**: `bridge-vortex-risk-assessment`
   - **Description**: `桥梁涡振风险评估模型 - SRTP项目`
   - **Visibility**: 选择 `Private`（推荐）或 `Public`
   - **Initialize**: 不要勾选任何选项（因为我们已有文件）
3. 点击 "Create repository"

#### 1.3 获取仓库地址
创建完成后，复制仓库的HTTPS地址，格式类似：
```
https://github.com/你的用户名/bridge-vortex-risk-assessment.git
```

### 第二步：上传项目到GitHub

#### 2.1 安装Git（如果未安装）
- 下载地址：[git-scm.com](https://git-scm.com/)
- 安装完成后，重启命令行

#### 2.2 配置Git（首次使用）
```bash
git config --global user.name "你的姓名"
git config --global user.email "你的邮箱@example.com"
```

#### 2.3 初始化并上传项目
在项目根目录 `d:/Desktop/SRTP` 打开命令行，执行：

```bash
# 初始化Git仓库
git init

# 添加所有文件
git add .

# 创建首次提交
git commit -m "初始化桥梁涡振风险评估项目"

# 连接远程仓库
git remote add origin https://github.com/你的用户名/bridge-vortex-risk-assessment.git

# 推送到GitHub
git push -u origin main
```

### 第三步：邀请组员

#### 3.1 添加协作者
1. 在GitHub仓库页面，点击 "Settings"
2. 左侧菜单选择 "Collaborators"
3. 点击 "Add people"
4. 输入组员的GitHub用户名或邮箱
5. 选择权限级别：
   - **Write**: 推荐，可以推送代码和管理issues
   - **Admin**: 完全权限（谨慎使用）

#### 3.2 发送邀请
- 组员会收到邮件邀请
- 或者直接分享仓库链接给组员

---

## 👥 组员加入项目指南

### 第一步：接受邀请
1. 检查邮箱，点击GitHub邀请链接
2. 或直接访问项目仓库页面，点击 "Accept invitation"

### 第二步：克隆项目到本地

#### 2.1 选择工作目录
在你希望存放项目的位置（如 `D:/Projects/`）打开命令行

#### 2.2 克隆仓库
```bash
git clone https://github.com/项目负责人用户名/bridge-vortex-risk-assessment.git
cd bridge-vortex-risk-assessment
```

#### 2.3 验证项目结构
确认你看到以下文件结构：
```
bridge-vortex-risk-assessment/
├── README.md
├── data/
├── docs/
├── src/
├── requirements.txt
└── 大概思路.md
```

---

## 🔄 日常协作流程

### 工作前：同步最新代码
```bash
# 拉取最新代码
git pull origin main
```

### 工作中：提交你的更改

#### 1. 查看文件状态
```bash
git status
```

#### 2. 添加修改的文件
```bash
# 添加特定文件
git add data/raw/新论文数据.csv
git add docs/项目进度跟踪.md

# 或添加所有修改
git add .
```

#### 3. 提交更改
```bash
git commit -m "描述你的更改内容"
```

**提交信息示例：**
- `"添加5篇桥梁涡振论文数据"`
- `"更新个人项目进度 - 完成第一周数据收集"`
- `"上传润扬大桥涡振分析PDF文件"`
- `"修复数据模板中的单位错误"`

#### 4. 推送到GitHub
```bash
git push origin main
```

### 工作后：检查推送结果
- 访问GitHub仓库页面
- 确认你的文件已成功上传
- 检查提交历史

---

## 📁 文件管理规范

### 目录结构说明
```
bridge-vortex-risk-assessment/
├── data/
│   ├── raw/                    # 原始数据文件
│   │   ├── papers/            # PDF论文文件
│   │   └── extracted/         # 提取的CSV数据
│   ├── processed/             # 处理后的数据
│   └── templates/             # 数据模板和指南
├── docs/
│   ├── 项目进度跟踪.md         # 项目总体进度
│   ├── 个人进度/              # 个人进度文件夹
│   │   ├── 张三_进度.md
│   │   ├── 李四_进度.md
│   │   └── 王五_进度.md
│   └── 会议记录/              # 会议纪要
├── src/                       # 源代码
├── results/                   # 结果文件
└── README.md
```

### 文件命名规范

#### PDF文件命名
```
论文标题_第一作者_年份.pdf
例如：大跨度悬索桥涡振响应分析_张三_2023.pdf
```

#### CSV数据文件命名
```
桥梁名称_数据类型_日期.csv
例如：润扬大桥_涡振数据_20241201.csv
```

#### 个人进度文件命名
```
姓名_进度_YYYYMMDD.md
例如：张三_进度_20241201.md
```

### 添加新文件的步骤

#### 1. 添加PDF论文
```bash
# 将PDF文件放入正确目录
cp "新论文.pdf" data/raw/papers/

# 重命名为规范格式
mv data/raw/papers/新论文.pdf data/raw/papers/桥梁涡振分析_李明_2023.pdf

# 提交
git add data/raw/papers/桥梁涡振分析_李明_2023.pdf
git commit -m "添加桥梁涡振分析论文 - 李明2023"
git push origin main
```

#### 2. 添加数据文件
```bash
# 将CSV文件放入正确目录
cp "提取数据.csv" data/raw/extracted/

# 重命名
mv data/raw/extracted/提取数据.csv data/raw/extracted/苏通大桥_涡振数据_20241201.csv

# 提交
git add data/raw/extracted/苏通大桥_涡振数据_20241201.csv
git commit -m "添加苏通大桥涡振数据"
git push origin main
```

#### 3. 更新个人进度
```bash
# 编辑你的进度文件
# 文件路径：docs/个人进度/你的姓名_进度.md

# 提交更新
git add docs/个人进度/你的姓名_进度.md
git commit -m "更新个人进度 - 完成第二周数据收集"
git push origin main
```

---

## ❗ 常见问题解决

### 问题1：推送时出现冲突
**现象：** `error: failed to push some refs`

**解决方案：**
```bash
# 先拉取最新代码
git pull origin main

# 如果有冲突，手动解决后再提交
git add .
git commit -m "解决合并冲突"
git push origin main
```

### 问题2：忘记拉取最新代码就开始工作
**解决方案：**
```bash
# 暂存当前工作
git stash

# 拉取最新代码
git pull origin main

# 恢复你的工作
git stash pop
```

### 问题3：误提交了错误文件
**解决方案：**
```bash
# 撤销最后一次提交（保留文件修改）
git reset --soft HEAD~1

# 重新添加正确的文件
git add 正确的文件
git commit -m "正确的提交信息"
```

### 问题4：大文件上传失败
**现象：** 文件超过100MB

**解决方案：**
1. 使用Git LFS（大文件存储）
2. 或将大文件存储在云盘，在README中提供链接

---

## 🏆 协作最佳实践

### 1. 提交频率
- **推荐：** 每天至少提交一次
- **最佳：** 完成一个小任务就提交一次
- **避免：** 积累大量修改后一次性提交

### 2. 提交信息规范
- **好的示例：**
  - `"添加港珠澳大桥涡振数据 - 包含5个测点"`
  - `"修复数据模板中频率单位错误"`
  - `"更新项目进度 - 完成文献调研阶段"`

- **避免的示例：**
  - `"更新"`
  - `"修改文件"`
  - `"临时保存"`

### 3. 分工协作建议
- **数据收集：** 按桥梁类型或地区分工
- **代码开发：** 按功能模块分工
- **文档维护：** 指定专人负责
- **进度跟踪：** 每周统一更新

### 4. 沟通机制
- **每周例会：** 同步进度，解决问题
- **GitHub Issues：** 记录问题和任务
- **微信群：** 日常快速沟通
- **项目文档：** 重要决策记录

### 5. 数据质量控制
- **双人检查：** 重要数据由两人验证
- **版本控制：** 重要修改前先备份
- **定期备份：** 每周备份到云盘

---

## 📞 技术支持

### 遇到Git问题时
1. 先查看本文档的常见问题部分
2. 搜索错误信息：`git 错误信息 解决方案`
3. 向项目负责人求助
4. 参考官方文档：[git-scm.com/docs](https://git-scm.com/docs)

### 联系方式
- **项目负责人：** [你的联系方式]
- **技术支持：** [技术负责人联系方式]
- **紧急情况：** [紧急联系方式]

---

## 🎉 开始协作

现在你已经掌握了GitHub协作的所有要点！

**立即行动清单：**
- [ ] 项目负责人：创建GitHub仓库并上传项目
- [ ] 项目负责人：邀请所有组员
- [ ] 组员：接受邀请并克隆项目
- [ ] 所有人：创建个人进度文件
- [ ] 所有人：测试提交和推送流程

**记住：** 
- 🔄 工作前先 `git pull`
- 💾 工作后及时 `git push`
- 📝 写清楚提交信息
- 🤝 遇到问题及时沟通

祝项目协作顺利！🚀