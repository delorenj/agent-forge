"""
Comprehensive test runner for AgentForge system.

Orchestrates all tests and provides detailed reporting of test results,
coverage, and system validation.
"""

import asyncio
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
import subprocess
import json


class AgentForgeTestRunner:
    """Comprehensive test runner for the AgentForge system."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        self.total_tests_run = 0
        self.total_tests_passed = 0
        self.total_tests_failed = 0
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all AgentForge tests and return comprehensive results."""
        print("🔥 AgentForge Comprehensive Test Suite")
        print("=" * 60)
        print("Testing the complete meta-agent orchestration system")
        print("")
        
        self.start_time = time.time()
        
        # Test suite configuration
        test_suites = [
            {
                "name": "Unit Tests - Engineering Manager",
                "module": "test_engineering_manager",
                "description": "Core orchestration and workflow management",
                "critical": True
            },
            {
                "name": "Unit Tests - Systems Analyst", 
                "module": "test_systems_analyst",
                "description": "Goal analysis and strategy generation",
                "critical": True
            },
            {
                "name": "Integration Tests",
                "module": "test_integration", 
                "description": "Agent communication and coordination",
                "critical": True
            },
            {
                "name": "End-to-End Tests",
                "module": "test_end_to_end",
                "description": "Complete workflow validation",
                "critical": True
            },
            {
                "name": "Performance Tests",
                "module": "test_performance",
                "description": "Performance and scalability validation", 
                "critical": False
            }
        ]
        
        # Run each test suite
        for suite_config in test_suites:
            await self._run_test_suite(suite_config)
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        return self._generate_final_report()
    
    async def _run_test_suite(self, suite_config: Dict[str, Any]):
        """Run a specific test suite."""
        suite_name = suite_config["name"]
        module_name = suite_config["module"]
        description = suite_config["description"]
        is_critical = suite_config["critical"]
        
        print(f"📋 {suite_name}")
        print(f"   {description}")
        print(f"   Module: {module_name}")
        
        try:
            # Run pytest on the specific module
            result = await self._run_pytest(module_name)
            
            self.test_results[suite_name] = {
                "module": module_name,
                "description": description,
                "critical": is_critical,
                "status": "passed" if result["return_code"] == 0 else "failed",
                "tests_run": result.get("tests_collected", 0),
                "tests_passed": result.get("tests_passed", 0),
                "tests_failed": result.get("tests_failed", 0),
                "duration": result.get("duration", 0),
                "output": result.get("output", ""),
                "errors": result.get("errors", [])
            }
            
            # Update totals
            self.total_tests_run += result.get("tests_collected", 0)
            self.total_tests_passed += result.get("tests_passed", 0) 
            self.total_tests_failed += result.get("tests_failed", 0)
            
            if result["return_code"] == 0:
                print(f"   ✅ PASSED ({result.get('tests_passed', 0)} tests)")
            else:
                print(f"   ❌ FAILED ({result.get('tests_failed', 0)} failures)")
                if is_critical:
                    print(f"   🚨 CRITICAL FAILURE - System may not be functional")
        
        except Exception as e:
            self.test_results[suite_name] = {
                "module": module_name,
                "description": description,
                "critical": is_critical,
                "status": "error",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 1,
                "duration": 0,
                "output": "",
                "errors": [str(e)]
            }
            print(f"   💥 ERROR: {str(e)}")
            
        print("")
    
    async def _run_pytest(self, module_name: str) -> Dict[str, Any]:
        """Run pytest on a specific module and parse results."""
        test_file = f"tests/{module_name}.py"
        
        # Check if test file exists
        if not Path(test_file).exists():
            return {
                "return_code": 1,
                "output": f"Test file {test_file} not found",
                "errors": [f"Missing test file: {test_file}"],
                "tests_collected": 0,
                "tests_passed": 0,
                "tests_failed": 1,
                "duration": 0
            }
        
        # Run pytest with JSON output
        cmd = [
            sys.executable, "-m", "pytest", 
            test_file,
            "--tb=short",
            "--maxfail=10",  # Stop after 10 failures
            "-v",  # Verbose output
            "--disable-warnings"  # Reduce noise
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5-minute timeout per test suite
            )
            duration = time.time() - start_time
            
            # Parse pytest output
            output_lines = result.stdout.split('\n')
            error_lines = result.stderr.split('\n')
            
            # Extract test statistics
            tests_collected = self._extract_pytest_stat(output_lines, "collected")
            tests_passed = self._extract_pytest_stat(output_lines, "passed")
            tests_failed = self._extract_pytest_stat(output_lines, "failed")
            tests_error = self._extract_pytest_stat(output_lines, "error")
            
            return {
                "return_code": result.returncode,
                "output": result.stdout,
                "stderr": result.stderr,
                "tests_collected": tests_collected,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed + tests_error,
                "duration": duration,
                "errors": [line for line in error_lines if line.strip()]
            }
            
        except subprocess.TimeoutExpired:
            return {
                "return_code": 1,
                "output": "Test suite timed out",
                "errors": ["Test execution timeout (5 minutes)"],
                "tests_collected": 0,
                "tests_passed": 0, 
                "tests_failed": 1,
                "duration": 300
            }
        except Exception as e:
            return {
                "return_code": 1,
                "output": f"Failed to run pytest: {str(e)}",
                "errors": [str(e)],
                "tests_collected": 0,
                "tests_passed": 0,
                "tests_failed": 1,
                "duration": 0
            }
    
    def _extract_pytest_stat(self, lines: List[str], stat_type: str) -> int:
        """Extract test statistics from pytest output."""
        for line in reversed(lines):  # Check from end first
            if stat_type in line.lower():
                # Look for patterns like "5 passed", "2 failed", etc.
                words = line.split()
                for i, word in enumerate(words):
                    if stat_type in word.lower() and i > 0:
                        try:
                            return int(words[i-1])
                        except (ValueError, IndexError):
                            continue
        return 0
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final test report."""
        total_duration = self.end_time - self.start_time if self.end_time and self.start_time else 0
        
        # Calculate summary statistics
        critical_failures = sum(1 for result in self.test_results.values() 
                               if result["critical"] and result["status"] != "passed")
        
        all_passed = all(result["status"] == "passed" for result in self.test_results.values())
        critical_passed = all(result["status"] == "passed" for result in self.test_results.values() 
                             if result["critical"])
        
        # Generate status assessment
        if all_passed:
            overall_status = "EXCELLENT"
            status_emoji = "🎉"
            status_message = "All tests passed! AgentForge is fully operational."
        elif critical_passed:
            overall_status = "GOOD"
            status_emoji = "✅"
            status_message = "Critical tests passed. AgentForge core functionality is operational."
        elif critical_failures == 0:
            overall_status = "WARNING"
            status_emoji = "⚠️"
            status_message = "Some non-critical tests failed. Core functionality should work."
        else:
            overall_status = "CRITICAL"
            status_emoji = "🚨"
            status_message = "Critical tests failed. AgentForge may not function properly."
        
        report = {
            "summary": {
                "overall_status": overall_status,
                "status_emoji": status_emoji,
                "status_message": status_message,
                "total_duration": total_duration,
                "total_test_suites": len(self.test_results),
                "total_tests_run": self.total_tests_run,
                "total_tests_passed": self.total_tests_passed,
                "total_tests_failed": self.total_tests_failed,
                "pass_rate": (self.total_tests_passed / self.total_tests_run * 100) if self.total_tests_run > 0 else 0,
                "critical_failures": critical_failures
            },
            "test_suites": self.test_results,
            "recommendations": self._generate_recommendations()
        }
        
        self._print_final_report(report)
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        failed_suites = [name for name, result in self.test_results.items() 
                        if result["status"] != "passed"]
        
        critical_failed = [name for name, result in self.test_results.items() 
                          if result["critical"] and result["status"] != "passed"]
        
        if critical_failed:
            recommendations.append("🚨 URGENT: Fix critical test failures before using AgentForge in production")
            for suite_name in critical_failed:
                recommendations.append(f"   - Review and fix failures in {suite_name}")
        
        if failed_suites and not critical_failed:
            recommendations.append("⚠️ Address non-critical test failures to improve system reliability")
        
        if self.total_tests_run == 0:
            recommendations.append("❌ No tests were executed - check test configuration and dependencies")
        
        if len(self.test_results) < 4:
            recommendations.append("📝 Consider adding more comprehensive test coverage")
        
        if not recommendations:
            recommendations.extend([
                "🎯 All tests passed! AgentForge is ready for use",
                "💡 Consider running performance tests regularly to monitor system health",
                "📊 Add integration tests with real Agno agents for more thorough validation"
            ])
        
        return recommendations
    
    def _print_final_report(self, report: Dict[str, Any]):
        """Print formatted final report."""
        summary = report["summary"]
        
        print("=" * 60)
        print("🎯 AGENTFORGE TEST RESULTS SUMMARY")
        print("=" * 60)
        print()
        
        # Overall status
        print(f"{summary['status_emoji']} Overall Status: {summary['overall_status']}")
        print(f"   {summary['status_message']}")
        print()
        
        # Statistics
        print("📊 Test Statistics:")
        print(f"   Total Test Suites: {summary['total_test_suites']}")
        print(f"   Total Tests Run: {summary['total_tests_run']}")
        print(f"   Tests Passed: {summary['total_tests_passed']}")
        print(f"   Tests Failed: {summary['total_tests_failed']}")
        print(f"   Pass Rate: {summary['pass_rate']:.1f}%")
        print(f"   Execution Time: {summary['total_duration']:.1f} seconds")
        print()
        
        # Suite-by-suite breakdown
        print("📋 Test Suite Breakdown:")
        for suite_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] == "passed" else "❌"
            critical_icon = "🚨" if result["critical"] else "📝"
            
            print(f"   {status_icon} {critical_icon} {suite_name}")
            print(f"      Status: {result['status'].upper()}")
            print(f"      Tests: {result['tests_passed']}/{result['tests_run']} passed")
            print(f"      Duration: {result['duration']:.2f}s")
            
            if result["errors"]:
                print(f"      Errors: {len(result['errors'])}")
        print()
        
        # Recommendations
        print("💡 Recommendations:")
        for recommendation in report["recommendations"]:
            print(f"   {recommendation}")
        print()
        
        # Final verdict
        if summary["overall_status"] == "EXCELLENT":
            print("🎉 CONGRATULATIONS! AgentForge is fully tested and operational!")
        elif summary["overall_status"] == "GOOD":
            print("✅ AgentForge core functionality is validated and ready to use.")
        elif summary["overall_status"] == "WARNING":
            print("⚠️ AgentForge should work, but address the issues for better reliability.")
        else:
            print("🚨 AgentForge has critical issues - fix failures before production use.")
        
        print("=" * 60)


async def run_comprehensive_tests():
    """Run all AgentForge tests comprehensively."""
    runner = AgentForgeTestRunner()
    
    try:
        results = await runner.run_all_tests()
        
        # Save detailed results to file
        results_file = "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📄 Detailed results saved to: {results_file}")
        
        # Return exit code based on critical test results
        if results["summary"]["critical_failures"] > 0:
            return 1  # Critical failure
        elif results["summary"]["total_tests_failed"] > 0:
            return 2  # Non-critical failures
        else:
            return 0  # All tests passed
            
    except Exception as e:
        print(f"💥 Test runner failed with error: {str(e)}")
        traceback.print_exc()
        return 3  # Test runner error


def run_quick_validation():
    """Run a quick validation check of core functionality."""
    print("🚀 AgentForge Quick Validation")
    print("=" * 40)
    
    try:
        # Test imports
        print("📦 Testing imports...")
        from agents.engineering_manager import EngineeringManager, InputGoal, ComplexityLevel
        from agents.systems_analyst import SystemsAnalyst
        from agents.base import AgentForgeBase
        print("   ✅ Core imports successful")
        
        # Test basic instantiation
        print("🔧 Testing instantiation...")
        em = EngineeringManager()
        analyst = SystemsAnalyst()
        print("   ✅ Agent instantiation successful")
        
        # Test basic model validation
        print("📝 Testing data models...")
        goal = InputGoal(
            goal_description="Test goal",
            domain="test",
            complexity_level=ComplexityLevel.MEDIUM
        )
        print("   ✅ Pydantic models working")
        
        print()
        print("🎉 Quick validation PASSED!")
        print("   AgentForge core functionality is working")
        print("   Run full tests with: python tests/test_runner.py")
        return True
        
    except Exception as e:
        print(f"❌ Quick validation FAILED: {str(e)}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Run quick validation
        success = run_quick_validation()
        sys.exit(0 if success else 1)
    else:
        # Run comprehensive tests
        exit_code = asyncio.run(run_comprehensive_tests())
        sys.exit(exit_code)