// Copyright (c) OpenMMLab. All rights reserved

 /*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related 
 * documentation and any modifications thereto. Any use, reproduction, 
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or 
 * its affiliates is strictly prohibited.
 * This source code file is only allowed to repro exps of cp-cluster.
*/

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"

Tensor cp_cluster_cpu(Tensor boxes, Tensor scores, Tensor dets,
                      float iou_threshold, float min_score,
                      int offset, float wfa_thresh, int tune_coords, int opt_id) {
  /*
  ** Implementation of Confidence Propagation Bounding Boxes Cluster for Object Detectors.
  **
  ** -boxes: The pytorch tensor containing box coordinates in the format of [N, 4].
  ** -scores: The pytorch tensor containing scores in the format of [N,].
  ** -dets: The result pytorch tensor buffer in the format of [N, 5].
  ** -iou_threshold: The IOU threshold to construct the graphs.
  ** -min_score: The miminum confidence value to filter out too weak boxes.
  ** -offset: The offset used in IOU calculation, following the MMCV style.
  ** -wfa_thresh: The IOU threshold to distinguish weak friends.
  ** -tune_coords: Whether update coordinates of bboxes with positive messages.
  ** -opt_id: The config options(1-3), as corresponds to the Config1-Config3 described in the paper.
  **
  */
  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong));
  }

  auto x1_t = boxes.select(1, 0).contiguous();
  auto y1_t = boxes.select(1, 1).contiguous();
  auto x2_t = boxes.select(1, 2).contiguous();
  auto y2_t = boxes.select(1, 3).contiguous();
  auto scores_t = scores.clone();

  Tensor areas_t = (x2_t - x1_t + offset) * (y2_t - y1_t + offset);

  auto nboxes = boxes.size(0);
  auto x1 = x1_t.data_ptr<float>();
  auto y1 = y1_t.data_ptr<float>();
  auto x2 = x2_t.data_ptr<float>();
  auto y2 = y2_t.data_ptr<float>();
  auto sc = scores_t.data_ptr<float>();
  auto areas = areas_t.data_ptr<float>();
  auto de = dets.data_ptr<float>();

  // Suppress mat: [i, j]=m indicates that Det[i] suppressed Det[j] for m times.
  std::vector<float> suppress_mat(nboxes * nboxes, 0.0f);
  // Positive messages
  std::vector<float> positive_msgs(nboxes * 6, 0.0f);
  // Negative messages
  std::vector<float> negative_msgs(nboxes * 3, 0.0f);
  // Dynamic IOU thresholds for different iterations
  float iou_thresholds[10] = {0.0f};
  // Weights for positive and negative messages
  std::vector<float> alphas(10, 1.0f);
  std::vector<float> betas(10, 1.0f);
  // Strategies for negative messages
  std::vector<int> neg_strategies(10, 0); // 0 for strongest neighbors, 1 for closest neighbors.

  int64_t pos = 0;
  Tensor inds_t = at::arange(nboxes, boxes.options().dtype(at::kLong));
  auto inds = inds_t.data_ptr<int64_t>();

  // Set up main configs based on opt_id(config1, config2, config3, config4).
  // Config 4 will incorporate better AP50 but less improvement on AP75.
  // By default it's config1.
  int64_t opt_max_iter = 2;
  float max_suppress_time = 0.99f;// Suppression from boxA to boxB can only happen once.
  iou_thresholds[0] = iou_threshold;
  iou_thresholds[1] = iou_thresholds[0] + 0.1f;
  neg_strategies[0] = 0;
  neg_strategies[1] = 1;
  assert(opt_id == 1 || opt_id==2 || opt_id==3 || opt_id==4);
  switch(opt_id) {
    case 1:
      break;
    case 2:
      iou_thresholds[1] = iou_thresholds[0] + 0.2f;
      break;
    case 3:
      iou_thresholds[1] = iou_thresholds[0] + 0.2f;
      max_suppress_time = 1.99f; // Suppression from boxA to boxB can happen twice.
      break;
    case 4:
      max_suppress_time = 1.99f; // Suppression from boxA to boxB can happen twice.
      break;
  }

  for (int64_t iter = 0; iter < opt_max_iter; iter++) {
    std::fill(positive_msgs.begin(), positive_msgs.end(), 0.0f);
    std::fill(negative_msgs.begin(), negative_msgs.end(), 0.0f);
    for (int64_t i = 0; i < nboxes; i++) {
      auto ix1 = de[i * 5 + 0] = x1[i];
      auto iy1 = de[i * 5 + 1] = y1[i];
      auto ix2 = de[i * 5 + 2] = x2[i];
      auto iy2 = de[i * 5 + 3] = y2[i];
      auto iscore = de[i * 5 + 4] = sc[i];
      auto iarea = areas[i];
      if (iscore < min_score)
        continue;

      for (pos = 0; pos < nboxes; pos++) {
        auto xx1 = std::max(ix1, x1[pos]);
        auto yy1 = std::max(iy1, y1[pos]);
        auto xx2 = std::min(ix2, x2[pos]);
        auto yy2 = std::min(iy2, y2[pos]);
        auto score = sc[pos];

        if (pos == i || score < min_score || score > iscore)
          continue;

        auto w = std::max(0.f, xx2 - xx1 + offset);
        auto h = std::max(0.f, yy2 - yy1 + offset);
        auto inter = w * h;
        auto ovr = inter / (iarea + areas[pos] - inter);

        if (ovr > iou_thresholds[iter]) {
          // Update negative messages(stronger box -> weaker box).
          if (neg_strategies[iter] == 0) {
            if (suppress_mat[i*nboxes + pos] < max_suppress_time && iscore > negative_msgs[3*pos + 1]) {
              negative_msgs[3*pos + 0] = ovr;
              negative_msgs[3*pos + 1] = iscore;
              negative_msgs[3*pos + 2] = static_cast<float>(i);
            }
          } else if (neg_strategies[iter] == 1) {
            if (suppress_mat[i*nboxes + pos] < max_suppress_time && ovr > negative_msgs[3*pos + 0]) {
              negative_msgs[3*pos + 0] = ovr;
              negative_msgs[3*pos + 1] = iscore;
              negative_msgs[3*pos + 2] = static_cast<float>(i);
            }
          }
        }

        // Update positive messages.
        if (ovr >= wfa_thresh) {
          // Update positive messages(weaker box -> stronger box).
          positive_msgs[6*i + 1] = positive_msgs[6*i + 1] + 1.0;
          if (sc[pos] > positive_msgs[6*i + 0]) {
            positive_msgs[6*i + 0] = sc[pos];
            positive_msgs[6*i + 2] = x1[pos];
            positive_msgs[6*i + 3] = y1[pos];
            positive_msgs[6*i + 4] = x2[pos];
            positive_msgs[6*i + 5] = y2[pos];
            //positive_msgs[6*i + 2] = positive_msgs[6*i + 2] + (x1[pos] - ix1);
            //positive_msgs[6*i + 3] = positive_msgs[6*i + 3] + (y1[pos] - iy1);
            //positive_msgs[6*i + 4] = positive_msgs[6*i + 4] + (x2[pos] - ix2);
            //positive_msgs[6*i + 5] = positive_msgs[6*i + 5] + (y2[pos] - iy2);
          }
        }
      }
    }
    for (int64_t i = 0; i < nboxes; i++) {
      sc[i] = sc[i] + alphas[iter] * (1.0 - sc[i]) * (positive_msgs[6*i + 1] / (positive_msgs[6*i + 1] + 1.0)) * positive_msgs[6*i + 0];
      //x1[i] = x1[i] + positive_msgs[6*i + 2] / (positive_msgs[6*i + 1] + 0.01);
      //y1[i] = y1[i] + positive_msgs[6*i + 3] / (positive_msgs[6*i + 1] + 0.01);
      //x2[i] = x2[i] + positive_msgs[6*i + 4] / (positive_msgs[6*i + 1] + 0.01);
      //y2[i] = y2[i] + positive_msgs[6*i + 5] / (positive_msgs[6*i + 1] + 0.01);
      if (tune_coords == 1) {
        x1[i] = (x1[i] * sc[i] + positive_msgs[6*i + 0] * positive_msgs[6*i + 2]) / (sc[i] + positive_msgs[6*i + 0]);
        y1[i] = (y1[i] * sc[i] + positive_msgs[6*i + 0] * positive_msgs[6*i + 3]) / (sc[i] + positive_msgs[6*i + 0]);
        x2[i] = (x2[i] * sc[i] + positive_msgs[6*i + 0] * positive_msgs[6*i + 4]) / (sc[i] + positive_msgs[6*i + 0]);
        y2[i] = (y2[i] * sc[i] + positive_msgs[6*i + 0] * positive_msgs[6*i + 5]) / (sc[i] + positive_msgs[6*i + 0]);
      }
      if (negative_msgs[3*i + 0] > 0.01f) {
        sc[i] = sc[i] - betas[iter] * sc[i] * negative_msgs[3*i + 0];
        auto idx_suppress = static_cast<int64_t>(negative_msgs[3*i + 2]);
        suppress_mat[nboxes * idx_suppress + i] = suppress_mat[nboxes * idx_suppress + i] + 1.0f;
      }
      //de[i * 5 + 0] = x1[i];
      //de[i * 5 + 1] = y1[i];
      //de[i * 5 + 2] = x2[i];
      //de[i * 5 + 3] = y2[i];
      de[i * 5 + 4] = sc[i];
    }
  }

  // To meet the mmcv and torchvision NMS API standard,
  // we sort the bounding boxes in descending order and filter out low conf boxes with too low scores.
  for (int64_t i = 0; i < nboxes; i++) {
    auto max_score = sc[i];
    auto max_pos = i;

    pos = i + 1;
    // get max box
    while (pos < nboxes) {
      if (max_score < sc[pos]) {
        max_score = sc[pos];
        max_pos = pos;
      }
      pos = pos + 1;
    }

    // Check whether max conf is smaller than min score.
    if (max_score < min_score) {
      nboxes = i;
      break;
    }
    // swap
    auto ix1 = de[i * 5 + 0] = x1[max_pos];
    auto iy1 = de[i * 5 + 1] = y1[max_pos];
    auto ix2 = de[i * 5 + 2] = x2[max_pos];
    auto iy2 = de[i * 5 + 3] = y2[max_pos];
    auto iscore = de[i * 5 + 4] = sc[max_pos];
    auto iarea = areas[max_pos];
    auto iind = inds[max_pos];
    x1[max_pos] = x1[i];
    y1[max_pos] = y1[i];
    x2[max_pos] = x2[i];
    y2[max_pos] = y2[i];
    sc[max_pos] = sc[i];
    areas[max_pos] = areas[i];
    inds[max_pos] = inds[i];
    x1[i] = ix1;
    y1[i] = iy1;
    x2[i] = ix2;
    y2[i] = iy2;
    sc[i] = iscore;
    areas[i] = iarea;
    inds[i] = iind;
  }

  return inds_t.slice(0, 0, nboxes);
}

Tensor cp_cluster_impl(Tensor boxes, Tensor scores, Tensor dets,
                       float iou_threshold, float min_score,
                       int offset, float wfa_thresh, int tune_coords, int opt_id);
REGISTER_DEVICE_IMPL(cp_cluster_impl, CPU, cp_cluster_cpu);



Tensor nms_cpu(Tensor boxes, Tensor scores, float iou_threshold, int offset) {
  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong));
  }
  auto x1_t = boxes.select(1, 0).contiguous();
  auto y1_t = boxes.select(1, 1).contiguous();
  auto x2_t = boxes.select(1, 2).contiguous();
  auto y2_t = boxes.select(1, 3).contiguous();

  Tensor areas_t = (x2_t - x1_t + offset) * (y2_t - y1_t + offset);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto nboxes = boxes.size(0);
  Tensor select_t = at::ones({nboxes}, boxes.options().dtype(at::kBool));

  auto select = select_t.data_ptr<bool>();
  auto order = order_t.data_ptr<int64_t>();
  auto x1 = x1_t.data_ptr<float>();
  auto y1 = y1_t.data_ptr<float>();
  auto x2 = x2_t.data_ptr<float>();
  auto y2 = y2_t.data_ptr<float>();
  auto areas = areas_t.data_ptr<float>();

  for (int64_t _i = 0; _i < nboxes; _i++) {
    if (select[_i] == false) continue;
    auto i = order[_i];
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < nboxes; _j++) {
      if (select[_j] == false) continue;
      auto j = order[_j];
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(0.f, xx2 - xx1 + offset);
      auto h = std::max(0.f, yy2 - yy1 + offset);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr > iou_threshold) select[_j] = false;
    }
  }
  return order_t.masked_select(select_t);
}

Tensor nms_impl(Tensor boxes, Tensor scores, float iou_threshold, int offset);
REGISTER_DEVICE_IMPL(nms_impl, CPU, nms_cpu);

Tensor softnms_cpu(Tensor boxes, Tensor scores, Tensor dets,
                   float iou_threshold, float sigma, float min_score,
                   int method, int offset) {
  if (boxes.numel() == 0) {
    return at::empty({0}, boxes.options().dtype(at::kLong));
  }

  auto x1_t = boxes.select(1, 0).contiguous();
  auto y1_t = boxes.select(1, 1).contiguous();
  auto x2_t = boxes.select(1, 2).contiguous();
  auto y2_t = boxes.select(1, 3).contiguous();
  auto scores_t = scores.clone();

  Tensor areas_t = (x2_t - x1_t + offset) * (y2_t - y1_t + offset);

  auto nboxes = boxes.size(0);
  auto x1 = x1_t.data_ptr<float>();
  auto y1 = y1_t.data_ptr<float>();
  auto x2 = x2_t.data_ptr<float>();
  auto y2 = y2_t.data_ptr<float>();
  auto sc = scores_t.data_ptr<float>();
  auto areas = areas_t.data_ptr<float>();
  auto de = dets.data_ptr<float>();

  int64_t pos = 0;
  Tensor inds_t = at::arange(nboxes, boxes.options().dtype(at::kLong));
  auto inds = inds_t.data_ptr<int64_t>();

  for (int64_t i = 0; i < nboxes; i++) {
    auto max_score = sc[i];
    auto max_pos = i;

    pos = i + 1;
    // get max box
    while (pos < nboxes) {
      if (max_score < sc[pos]) {
        max_score = sc[pos];
        max_pos = pos;
      }
      pos = pos + 1;
    }
    // swap
    auto ix1 = de[i * 5 + 0] = x1[max_pos];
    auto iy1 = de[i * 5 + 1] = y1[max_pos];
    auto ix2 = de[i * 5 + 2] = x2[max_pos];
    auto iy2 = de[i * 5 + 3] = y2[max_pos];
    auto iscore = de[i * 5 + 4] = sc[max_pos];
    auto iarea = areas[max_pos];
    auto iind = inds[max_pos];
    x1[max_pos] = x1[i];
    y1[max_pos] = y1[i];
    x2[max_pos] = x2[i];
    y2[max_pos] = y2[i];
    sc[max_pos] = sc[i];
    areas[max_pos] = areas[i];
    inds[max_pos] = inds[i];
    x1[i] = ix1;
    y1[i] = iy1;
    x2[i] = ix2;
    y2[i] = iy2;
    sc[i] = iscore;
    areas[i] = iarea;
    inds[i] = iind;

    pos = i + 1;
    while (pos < nboxes) {
      auto xx1 = std::max(ix1, x1[pos]);
      auto yy1 = std::max(iy1, y1[pos]);
      auto xx2 = std::min(ix2, x2[pos]);
      auto yy2 = std::min(iy2, y2[pos]);

      auto w = std::max(0.f, xx2 - xx1 + offset);
      auto h = std::max(0.f, yy2 - yy1 + offset);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[pos] - inter);

      float weight = 1.;
      if (method == 0) {
        if (ovr >= iou_threshold) weight = 0;
      } else if (method == 1) {
        if (ovr >= iou_threshold) weight = 1 - ovr;
      } else if (method == 2) {
        weight = std::exp(-(ovr * ovr) / sigma);
      }
      sc[pos] *= weight;
      // if box score falls below threshold, discard the box by
      // swapping with last box update N
      if (sc[pos] < min_score) {
        x1[pos] = x1[nboxes - 1];
        y1[pos] = y1[nboxes - 1];
        x2[pos] = x2[nboxes - 1];
        y2[pos] = y2[nboxes - 1];
        sc[pos] = sc[nboxes - 1];
        areas[pos] = areas[nboxes - 1];
        inds[pos] = inds[nboxes - 1];
        nboxes = nboxes - 1;
        pos = pos - 1;
      }
      pos = pos + 1;
    }
  }
  return inds_t.slice(0, 0, nboxes);
}

Tensor softnms_impl(Tensor boxes, Tensor scores, Tensor dets,
                    float iou_threshold, float sigma, float min_score,
                    int method, int offset);
REGISTER_DEVICE_IMPL(softnms_impl, CPU, softnms_cpu);

std::vector<std::vector<int> > nms_match_cpu(Tensor dets, float iou_threshold) {
  auto x1_t = dets.select(1, 0).contiguous();
  auto y1_t = dets.select(1, 1).contiguous();
  auto x2_t = dets.select(1, 2).contiguous();
  auto y2_t = dets.select(1, 3).contiguous();
  auto scores = dets.select(1, 4).contiguous();

  at::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

  auto order_t = std::get<1>(scores.sort(0, /* descending=*/true));

  auto ndets = dets.size(0);
  at::Tensor suppressed_t =
      at::zeros({ndets}, dets.options().dtype(at::kByte).device(at::kCPU));

  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  auto order = order_t.data_ptr<int64_t>();
  auto x1 = x1_t.data_ptr<float>();
  auto y1 = y1_t.data_ptr<float>();
  auto x2 = x2_t.data_ptr<float>();
  auto y2 = y2_t.data_ptr<float>();
  auto areas = areas_t.data_ptr<float>();

  std::vector<int> keep;
  std::vector<std::vector<int> > matched;

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1) continue;
    keep.push_back(i);
    std::vector<int> v_i;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1) continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<float>(0), xx2 - xx1);
      auto h = std::max(static_cast<float>(0), yy2 - yy1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr >= iou_threshold) {
        suppressed[j] = 1;
        v_i.push_back(j);
      }
    }
    matched.push_back(v_i);
  }
  for (size_t i = 0; i < keep.size(); i++)
    matched[i].insert(matched[i].begin(), keep[i]);
  return matched;
}

std::vector<std::vector<int> > nms_match_impl(Tensor dets, float iou_threshold);
REGISTER_DEVICE_IMPL(nms_match_impl, CPU, nms_match_cpu);
