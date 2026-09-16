/** Method forms of the ops: `x.matmul(w).add(bias).softmax()`. Only ops that
 *  are Tensor methods in torch appear here.
 *
 *  A separate file so value.ts never imports ops.ts, which would close an
 *  import cycle; index.ts imports it for the side effect. */

import { Value } from './value.ts';
import {
  add,
  asinh,
  chunk,
  clamp,
  matmul,
  mean,
  mul,
  reshape,
  rsqrt,
  sigmoid,
  slice,
  softmax,
  split,
  sub,
  tanh,
  topk,
  transpose,
  type MatmulOpts,
} from './ops.ts';

Value.prototype.add = function (b) {
  return add(this, b);
};
Value.prototype.sub = function (b) {
  return sub(this, b);
};
Value.prototype.mul = function (b) {
  return mul(this, b);
};
Value.prototype.matmul = function (b, opts) {
  return matmul(this, b, opts);
};
Value.prototype.reshape = function (newDims) {
  return reshape(this, newDims);
};
Value.prototype.transpose = function () {
  return transpose(this);
};
Value.prototype.clamp = function (lo, hi) {
  return clamp(this, lo, hi);
};
Value.prototype.slice = function (dim, start, end) {
  return slice(this, dim, start, end);
};
Value.prototype.chunk = function (n, dim) {
  return chunk(this, n, dim);
};
Value.prototype.split = function (sizes, dim) {
  return split(this, sizes, dim);
};
Value.prototype.softmax = function () {
  return softmax(this);
};
Value.prototype.sigmoid = function () {
  return sigmoid(this);
};
Value.prototype.tanh = function () {
  return tanh(this);
};
Value.prototype.asinh = function () {
  return asinh(this);
};
Value.prototype.mean = function () {
  return mean(this);
};
Value.prototype.rsqrt = function () {
  return rsqrt(this);
};
Value.prototype.topk = function (k) {
  return topk(this, k);
};
