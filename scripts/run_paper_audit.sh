#!/bin/bash
# Paper Audit Tools Runner Script
# Based on workspace requirements for paper review tools

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    print_status "Using Python: $($PYTHON_CMD --version)"
}

# Check if required packages are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    $PYTHON_CMD -c "import yaml" 2>/dev/null || {
        print_warning "PyYAML not found. Installing..."
        $PYTHON_CMD -m pip install pyyaml
    }
    
    print_success "Dependencies check passed"
}

# Generate review reports from existing audit files
generate_reports() {
    print_status "Generating review reports from existing audit files..."
    
    if [ -f "scripts/generate_review_reports.py" ]; then
        $PYTHON_CMD scripts/generate_review_reports.py --workspace . --output-dir review_results
        print_success "Review reports generated successfully"
    else
        print_error "generate_review_reports.py not found"
        exit 1
    fi
}

# Run paper audit workflow on a specific paper
run_audit() {
    local paper_path="$1"
    local paper_title="$2"
    
    if [ -z "$paper_path" ] || [ -z "$paper_title" ]; then
        print_error "Usage: $0 audit <paper_path> <paper_title>"
        exit 1
    fi
    
    print_status "Running paper audit workflow..."
    print_status "Paper: $paper_title"
    print_status "Path: $paper_path"
    
    if [ -f "scripts/paper_audit_workflow.py" ]; then
        $PYTHON_CMD scripts/paper_audit_workflow.py --paper "$paper_path" --title "$paper_title"
        print_success "Paper audit completed"
    else
        print_error "paper_audit_workflow.py not found"
        exit 1
    fi
}

# Show help
show_help() {
    echo "Paper Audit Tools Runner"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  generate    Generate review reports from existing audit files"
    echo "  audit       Run paper audit workflow on a specific paper"
    echo "  check       Check dependencies and setup"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 generate"
    echo "  $0 audit paper/cache_audit/main.tex \"Cache Translation Across Heterogeneous LLMs\""
    echo "  $0 check"
}

# Main script logic
case "${1:-help}" in
    generate)
        check_python
        check_dependencies
        generate_reports
        ;;
    audit)
        check_python
        check_dependencies
        run_audit "$2" "$3"
        ;;
    check)
        check_python
        check_dependencies
        print_success "Setup check completed"
        ;;
    help|*)
        show_help
        ;;
esac