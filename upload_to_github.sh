#!/bin/bash
# =====================================================
# Triple-Hybrid RAG — GitHub 업로드 스크립트
# 실행: bash upload_to_github.sh
# =====================================================

REPO_URL="https://github.com/sdw1621/hybrid-rag-comparsion.git"
BRANCH="main"
COMMIT_MSG="feat: Triple-Hybrid RAG 전체 구현 (DWA + Evaluator + Streamlit + Colab)"

echo "🚀 GitHub 업로드 시작..."
echo "레포: $REPO_URL"
echo ""

# git 초기화 (이미 초기화된 경우 스킵)
if [ ! -d ".git" ]; then
    git init
    echo "✅ git init 완료"
fi

# 리모트 설정
git remote remove origin 2>/dev/null
git remote add origin $REPO_URL
echo "✅ remote 설정 완료"

# 브랜치 설정
git checkout -B $BRANCH 2>/dev/null || git checkout $BRANCH

# 스테이징 & 커밋
git add .
git status --short
echo ""
git commit -m "$COMMIT_MSG"

# 푸시
echo ""
echo "📤 push 중..."
git push -u origin $BRANCH --force

echo ""
echo "🎉 완료! 확인: $REPO_URL"
