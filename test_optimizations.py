#!/usr/bin/env python3
"""快速验证脚本：测试修改后的代码是否正常工作。"""

import sys
import numpy as np

def test_value_mapper():
    """测试ValueMapper类。"""
    print("Testing ValueMapper...")
    from apcs.mapper.math import ValueMapper
    
    # 创建测试数据
    rng = np.random.default_rng(42)
    L_t, L_s, S, H, D = 4, 3, 8, 4, 16
    kv_t = rng.standard_normal((L_t, S, H, D))
    kv_s = rng.standard_normal((L_s, S, H, D))
    
    # 创建简单层映射
    layer_map = [[0, 1], [1, 2], [2, 3]]
    
    # 测试ValueMapper
    mapper = ValueMapper(rank=8, n_iter=5)
    mapper.fit(kv_t, kv_s, layer_map, kv_kind="V")
    result = mapper.transform(kv_t, layer_map, kv_kind="V")
    
    assert result.shape == (L_s, S, H, D), f"Shape mismatch: {result.shape}"
    print("  ValueMapper OK")
    return True


def test_progressive_injection():
    """测试ProgressiveInjectionEvaluator类。"""
    print("Testing ProgressiveInjectionEvaluator...")
    from apcs.inference.evaluator import ProgressiveInjectionEvaluator, InjectionEvaluator
    
    # 创建mock evaluator
    class MockEvaluator:
        pass
    
    mock_eval = MockEvaluator()
    progressive_eval = ProgressiveInjectionEvaluator(
        base_evaluator=mock_eval,
        cka_threshold=0.3,
    )
    
    # 测试计算层兼容性
    rng = np.random.default_rng(42)
    teacher_kv = rng.standard_normal((4, 8, 4, 16))
    student_kv = rng.standard_normal((3, 8, 4, 16))
    
    compatibility = progressive_eval.compute_layer_compatibility(teacher_kv, student_kv)
    assert len(compatibility) == 3, f"Expected 3 layers, got {len(compatibility)}"
    
    # 测试选择可注入层
    injectable = progressive_eval.select_injectable_layers()
    assert isinstance(injectable, list), "injectable should be a list"
    
    print("  ProgressiveInjectionEvaluator OK")
    return True


def test_layer_alignment():
    """测试Layer Alignment优化。"""
    print("Testing Layer Alignment...")
    from apcs.alignment.runner import (
        select_layer_mapping,
        compare_alignment_strategies,
        evaluate_alignment_quality,
    )
    
    n_t, n_s = 4, 3
    rng = np.random.default_rng(42)
    sim_matrix = rng.uniform(0, 1, (n_s, n_t))
    
    # 测试select_layer_mapping
    for strategy in ["proportional", "last_layer", "data_driven_topk", "geometry_aware_topk"]:
        mapping = select_layer_mapping(strategy, n_t, n_s, sim_matrix)
        assert len(mapping) == n_s, f"Strategy {strategy} returned wrong number of layers"
    
    # 测试compare_alignment_strategies
    results = compare_alignment_strategies(n_t, n_s, sim_matrix)
    assert "proportional" in results
    assert "data_driven_topk" in results
    
    # 测试evaluate_alignment_quality
    quality = evaluate_alignment_quality(results["proportional"], sim_matrix)
    assert "avg_similarity" in quality
    
    print("  Layer Alignment OK")
    return True


def test_training_loss():
    """测试训练目标计算。"""
    print("Testing Training Loss...")
    from apcs.capability.main import compute_training_loss
    
    rng = np.random.default_rng(42)
    batch, seq, vocab = 2, 8, 100
    
    student_logits = rng.standard_normal((batch, seq, vocab))
    teacher_logits = rng.standard_normal((batch, seq, vocab))
    target_ids = rng.integers(0, vocab, (batch, seq))
    student_self_logits = rng.standard_normal((batch, seq, vocab))
    
    losses = compute_training_loss(
        student_logits, teacher_logits, target_ids, student_self_logits
    )
    
    assert "total_loss" in losses
    assert "l_task" in losses
    assert "l_self" in losses
    assert "l_teacher" in losses
    
    print("  Training Loss OK")
    return True


def test_ridge_per_head_mapper():
    """测试修复后的RidgePerHeadMapper。"""
    print("Testing RidgePerHeadMapper...")
    from apcs.mapper.math import RidgePerHeadMapper
    
    rng = np.random.default_rng(42)
    L_t, L_s, S, H, D = 4, 3, 8, 4, 16
    kv_t = rng.standard_normal((L_t, S, H, D))
    kv_s = rng.standard_normal((L_s, S, H, D))
    
    layer_map = [[0, 1], [1, 2], [2, 3]]
    
    mapper = RidgePerHeadMapper(lam=0.001)
    mapper.fit(kv_t, kv_s, layer_map, kv_kind="K")
    result = mapper.transform(kv_t, layer_map, kv_kind="K")
    
    assert result.shape == (L_s, S, H, D), f"Shape mismatch: {result.shape}"
    print("  RidgePerHeadMapper OK")
    return True


def main():
    """运行所有测试。"""
    tests = [
        test_value_mapper,
        test_progressive_injection,
        test_layer_alignment,
        test_training_loss,
        test_ridge_per_head_mapper,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
