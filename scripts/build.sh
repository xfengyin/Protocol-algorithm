#!/bin/bash
set -e

echo "🚀 Protocol-algorithm v2.0 构建脚本"

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查依赖
check_dependencies() {
    echo -e "${BLUE}检查依赖...${NC}"
    
    if ! command -v cargo &> /dev/null; then
        echo "❌ Rust 未安装，请访问 https://rustup.rs/ 安装"
        exit 1
    fi
    
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python3 未安装"
        exit 1
    fi
    
    if ! command -v npm &> /dev/null; then
        echo "❌ Node.js 未安装"
        exit 1
    fi
    
    echo -e "${GREEN}✓ 依赖检查通过${NC}"
}

# 构建 Rust 核心
build_core() {
    echo -e "${BLUE}构建 Rust 核心...${NC}"
    cd core
    cargo build --release
    cd ..
    echo -e "${GREEN}✓ Rust 核心构建完成${NC}"
}

# 构建 Python 绑定
build_python() {
    echo -e "${BLUE}构建 Python 绑定...${NC}"
    cd python
    pip install maturin
    maturin develop --release
    cd ..
    echo -e "${GREEN}✓ Python 绑定构建完成${NC}"
}

# 构建 Web 前端
build_web() {
    echo -e "${BLUE}构建 Web 前端...${NC}"
    cd web/frontend
    npm install
    npm run build
    cd ../..
    echo -e "${GREEN}✓ Web 前端构建完成${NC}"
}

# 运行测试
run_tests() {
    echo -e "${BLUE}运行测试...${NC}"
    cd core
    cargo test
    cd ../python
    pytest
    cd ..
    echo -e "${GREEN}✓ 测试完成${NC}"
}

# 主函数
main() {
    check_dependencies
    
    case "${1:-all}" in
        core)
            build_core
            ;;
        python)
            build_python
            ;;
        web)
            build_web
            ;;
        test)
            run_tests
            ;;
        all)
            build_core
            build_python
            build_web
            ;;
        *)
            echo "用法：$0 {core|python|web|test|all}"
            exit 1
            ;;
    esac
    
    echo -e "${GREEN}✅ 构建完成！${NC}"
}

main "$@"
