// Copyright 2021 Ivan Alles. See also the LICENSE file.

import * as tf from '@tensorflow/tfjs';
tf.setBackend('cpu')

import { testables } from '@/Engine'
const padSize = testables.padSize;
const makeInput = testables.makeInput;

describe.each([
  [1, 32, 31],
  [31, 32, 1],
  [32, 32, 0],
  [33, 32, 31],
  [63, 32, 1],
  [64, 32, 0],
])('padSize(%i, %i, %i)', (size, padTo, expected) => {
  const actual = padSize(size, padTo);

  test('result is as excpected', () => {
    expect(actual).toStrictEqual(expected);
  });
});

describe.each([
  [128, 128, 128, 32, 128, 128],
  [100, 128, 128, 32, 128, 128],
  [100, 51, 128, 32, 128, 64],
  [200, 150, 128, 32, 128, 96],
])('makeInput(%i, %i, %i, %i, %i, %i)', (width, height, maxSize, padTo, expectedWidth, expectedHeight) => {
  const image = tf.zeros([height, width, 3], 'float32');
  const input = makeInput(image, maxSize, padTo);

  test('input.image batch size is as excpected', () => {
    expect(input.image.shape[0]).toStrictEqual(1);
  });
  test('input.image height is as excpected', () => {
    expect(input.image.shape[1]).toStrictEqual(expectedHeight);
  });
  test('input.image width is as excpected', () => {
    expect(input.image.shape[2]).toStrictEqual(expectedWidth);
  });
  test('input.image channels are as excpected', () => {
    expect(input.image.shape[3]).toStrictEqual(3);
  });

});

