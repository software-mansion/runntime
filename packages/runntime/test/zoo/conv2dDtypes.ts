/** What the conv model tests read off a built model: the dtypes its
 *  nn.Conv2d layers load, and the dtypes of every other parameter. */
import { Conv2d } from '../../src/core/nn/layers.ts';
import type { AnyModule, Parameter } from '../../src/core/nn/module.ts';

/** The distinct dtypes of `model`'s parameters, split by whether an
 *  nn.Conv2d holds them. */
export function conv2dDtypes(model: AnyModule): { conv2d: string[]; other: string[] } {
  const inConv2d = new Set<Parameter>();
  for (const [, module] of model.namedModules()) {
    if (!(module instanceof Conv2d)) continue;
    for (const [, parameter] of module.namedParameters()) inConv2d.add(parameter);
  }
  const dtypes = (conv2d: boolean) =>
    [
      ...new Set(
        [...model.namedParameters()]
          .filter(([, parameter]) => inConv2d.has(parameter) === conv2d)
          .map(([, parameter]) => parameter.shape.dtype),
      ),
    ].sort();
  return { conv2d: dtypes(true), other: dtypes(false) };
}
