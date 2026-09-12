#!/usr/bin/env python3
"""
Review Report Generator
Generates comprehensive review reports from paper audit files

This script reads existing paper audit files and generates
structured review reports with revision suggestions.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import json


class ReviewReportGenerator:
    """Generate review reports from paper audit files."""
    
    def __init__(self, workspace_path: str = "."):
        """Initialize with workspace path."""
        self.workspace_path = Path(workspace_path)
        self.audit_files = []
        self.review_data = {}
        
    def find_audit_files(self) -> List[Path]:
        """Find all paper audit files in the workspace."""
        audit_pattern = "paper_audit*.md"
        audit_files = list(self.workspace_path.glob(audit_pattern))
        
        print(f"Found {len(audit_files)} audit files:")
        for f in audit_files:
            print(f"  - {f.name}")
        
        self.audit_files = audit_files
        return audit_files
    
    def parse_audit_file(self, file_path: Path) -> Dict[str, Any]:
        """Parse a paper audit file and extract structured data."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract title
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Unknown Paper"
        
        # Extract sections
        sections = {}
        current_section = None
        current_content = []
        
        for line in content.split('\n'):
            if line.startswith('## '):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line[3:].strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        # Extract major concerns
        major_concerns = []
        if '主要问题' in sections:
            concerns_text = sections['主要问题']
            # Extract numbered concerns
            concern_matches = re.findall(r'\*\*M\d+\.\s*(.+?)\*\*', concerns_text)
            major_concerns = concern_matches
        
        # Extract minor concerns
        minor_concerns = []
        if '次要问题' in sections:
            concerns_text = sections['次要问题']
            # Extract bullet points
            minor_matches = re.findall(r'-\s+(.+)', concerns_text)
            minor_concerns = minor_matches
        
        # Extract recommendation
        recommendation = ""
        if '审稿结论' in sections:
            rec_text = sections['审稿结论']
            # Extract recommendation
            rec_match = re.search(r'\*\*建议：(.+?)\*\*', rec_text)
            if rec_match:
                recommendation = rec_match.group(1)
        
        # Extract questions
        questions = []
        if '建议向作者提出的问题' in sections:
            questions_text = sections['建议向作者提出的问题']
            # Extract numbered questions
            question_matches = re.findall(r'\d+\.\s+(.+)', questions_text)
            questions = question_matches
        
        parsed_data = {
            'file_path': str(file_path),
            'title': title,
            'sections': sections,
            'major_concerns': major_concerns,
            'minor_concerns': minor_concerns,
            'recommendation': recommendation,
            'questions': questions,
            'parse_timestamp': datetime.now().isoformat()
        }
        
        return parsed_data
    
    def generate_executive_summary(self, all_reviews: List[Dict[str, Any]]) -> str:
        """Generate an executive summary from all reviews."""
        summary_lines = [
            "# Executive Summary: Paper Audit Results",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Reviews Analyzed:** {len(all_reviews)}",
            "\n## Overview\n"
        ]
        
        # Count recommendations
        recommendations = {}
        for review in all_reviews:
            rec = review.get('recommendation', 'Unknown')
            recommendations[rec] = recommendations.get(rec, 0) + 1
        
        summary_lines.append("### Recommendation Distribution")
        for rec, count in recommendations.items():
            summary_lines.append(f"- {rec}: {count} review(s)")
        
        # Collect all major concerns
        all_major_concerns = []
        for review in all_reviews:
            all_major_concerns.extend(review.get('major_concerns', []))
        
        summary_lines.append(f"\n### Major Concerns Summary")
        summary_lines.append(f"**Total Major Concerns:** {len(all_major_concerns)}")
        
        # Categorize concerns
        concern_categories = {}
        for concern in all_major_concerns:
            # Simple categorization based on keywords
            if 'empirical' in concern.lower() or 'experiment' in concern.lower():
                category = 'Empirical'
            elif 'method' in concern.lower():
                category = 'Methodology'
            elif 'claim' in concern.lower() or 'assertion' in concern.lower():
                category = 'Claims'
            elif 'writing' in concern.lower() or 'presentation' in concern.lower():
                category = 'Writing'
            else:
                category = 'Other'
            
            concern_categories[category] = concern_categories.get(category, 0) + 1
        
        for category, count in sorted(concern_categories.items(), key=lambda x: x[1], reverse=True):
            summary_lines.append(f"- {category}: {count} concerns")
        
        # Collect all questions
        all_questions = []
        for review in all_reviews:
            all_questions.extend(review.get('questions', []))
        
        summary_lines.append(f"\n### Questions for Authors")
        summary_lines.append(f"**Total Questions:** {len(all_questions)}")
        
        # Sample questions
        if all_questions:
            summary_lines.append("\n**Key Questions:**")
            for i, question in enumerate(all_questions[:5], 1):
                summary_lines.append(f"{i}. {question}")
        
        return "\n".join(summary_lines)
    
    def generate_revision_checklist(self, all_reviews: List[Dict[str, Any]]) -> str:
        """Generate a revision checklist from all reviews."""
        checklist_lines = [
            "# Revision Checklist",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Major Revisions (Must Address)\n"
        ]
        
        # Collect all major concerns
        major_items = []
        for review in all_reviews:
            for concern in review.get('major_concerns', []):
                major_items.append({
                    'description': concern,
                    'source': review.get('title', 'Unknown'),
                    'status': 'pending'
                })
        
        for i, item in enumerate(major_items, 1):
            checklist_lines.append(f"- [ ] {item['description']}")
            checklist_lines.append(f"  - Source: {item['source']}")
        
        checklist_lines.append("\n## Minor Revisions (Should Address)\n")
        
        # Collect all minor concerns
        minor_items = []
        for review in all_reviews:
            for concern in review.get('minor_concerns', []):
                minor_items.append({
                    'description': concern,
                    'source': review.get('title', 'Unknown'),
                    'status': 'pending'
                })
        
        for i, item in enumerate(minor_items, 1):
            checklist_lines.append(f"- [ ] {item['description']}")
            checklist_lines.append(f"  - Source: {item['source']}")
        
        checklist_lines.append("\n## Questions to Address\n")
        
        # Collect all questions
        all_questions = []
        for review in all_reviews:
            for question in review.get('questions', []):
                all_questions.append({
                    'question': question,
                    'source': review.get('title', 'Unknown')
                })
        
        for i, item in enumerate(all_questions, 1):
            checklist_lines.append(f"{i}. {item['question']}")
            checklist_lines.append(f"   - Source: {item['source']}")
        
        return "\n".join(checklist_lines)
    
    def generate_consolidated_report(self, output_dir: str = "review_results") -> Dict[str, Any]:
        """Generate a consolidated review report from all audit files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Find and parse all audit files
        audit_files = self.find_audit_files()
        all_reviews = []
        
        for audit_file in audit_files:
            print(f"\nParsing {audit_file.name}...")
            review_data = self.parse_audit_file(audit_file)
            all_reviews.append(review_data)
            
            # Save individual parsed review
            individual_output = output_path / f"parsed_{audit_file.stem}.json"
            with open(individual_output, 'w', encoding='utf-8') as f:
                json.dump(review_data, f, indent=2, ensure_ascii=False)
            print(f"  Saved parsed data to: {individual_output}")
        
        # Generate executive summary
        executive_summary = self.generate_executive_summary(all_reviews)
        summary_path = output_path / "executive_summary.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(executive_summary)
        print(f"\nGenerated executive summary: {summary_path}")
        
        # Generate revision checklist
        revision_checklist = self.generate_revision_checklist(all_reviews)
        checklist_path = output_path / "revision_checklist.md"
        with open(checklist_path, 'w', encoding='utf-8') as f:
            f.write(revision_checklist)
        print(f"Generated revision checklist: {checklist_path}")
        
        # Generate consolidated report
        consolidated_lines = [
            "# Consolidated Paper Audit Report",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Total Reviews:** {len(all_reviews)}",
            "\n## Individual Reviews\n"
        ]
        
        for i, review in enumerate(all_reviews, 1):
            consolidated_lines.append(f"### {i}. {review['title']}")
            consolidated_lines.append(f"- **File:** {review['file_path']}")
            consolidated_lines.append(f"- **Recommendation:** {review['recommendation']}")
            consolidated_lines.append(f"- **Major Concerns:** {len(review['major_concerns'])}")
            consolidated_lines.append(f"- **Minor Concerns:** {len(review['minor_concerns'])}")
            consolidated_lines.append(f"- **Questions:** {len(review['questions'])}")
            consolidated_lines.append("")
        
        consolidated_lines.append("\n## Key Findings\n")
        
        # Extract key findings across all reviews
        key_findings = []
        for review in all_reviews:
            for concern in review.get('major_concerns', []):
                if len(concern) > 50:  # Only substantial concerns
                    key_findings.append(concern)
        
        # Remove duplicates
        unique_findings = list(set(key_findings))
        
        for i, finding in enumerate(unique_findings[:10], 1):  # Top 10 findings
            consolidated_lines.append(f"{i}. {finding}")
        
        consolidated_report = "\n".join(consolidated_lines)
        report_path = output_path / "consolidated_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(consolidated_report)
        print(f"Generated consolidated report: {report_path}")
        
        # Summary statistics
        stats = {
            'total_reviews': len(all_reviews),
            'total_major_concerns': sum(len(r.get('major_concerns', [])) for r in all_reviews),
            'total_minor_concerns': sum(len(r.get('minor_concerns', [])) for r in all_reviews),
            'total_questions': sum(len(r.get('questions', [])) for r in all_reviews),
            'generated_files': [
                str(summary_path),
                str(checklist_path),
                str(report_path)
            ]
        }
        
        print(f"\n=== Generation Complete ===")
        print(f"Total reviews processed: {stats['total_reviews']}")
        print(f"Total major concerns: {stats['total_major_concerns']}")
        print(f"Total minor concerns: {stats['total_minor_concerns']}")
        print(f"Total questions: {stats['total_questions']}")
        
        return stats


def main():
    """Main entry point for the review report generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate review reports from paper audit files')
    parser.add_argument('--workspace', default='.', 
                       help='Workspace path (default: current directory)')
    parser.add_argument('--output-dir', default='review_results',
                       help='Output directory for reports (default: review_results)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = ReviewReportGenerator(args.workspace)
    
    # Generate consolidated report
    stats = generator.generate_consolidated_report(args.output_dir)
    
    print(f"\nAll reports generated in: {args.output_dir}/")


if __name__ == '__main__':
    main()