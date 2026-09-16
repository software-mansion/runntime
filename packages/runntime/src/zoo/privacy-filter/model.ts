/** The full privacy-filter network: a CPU-resident embedding table, N blocks
 *  of attention and MLP, a final RMSNorm, then the unembed. Parameter names
 *  match the checkpoint layout.
 *
 *  Rope tables are computed per call rather than stored, so one model instance
 *  serves every sequence length. */

import { nn, tensor, type Value } from '../../core/index.ts';
import { EagerAttnBlock, HEAD } from './attnBlock.ts';
import { EagerMoeMlp, type MoeDims } from './moeMlp.ts';
import { ropeConstants } from './rope.ts';

const HIDDEN = 640;
export { HIDDEN };
const RMS_EPS = 1e-5;

export class EagerBlock extends nn.Module<[Value, Value, Value], Value> {
  readonly attn: EagerAttnBlock;
  readonly mlp: EagerMoeMlp;

  constructor(moe: MoeDims) {
    super();
    this.attn = new EagerAttnBlock();
    this.mlp = new EagerMoeMlp(moe);
  }

  override forward(x: Value, cos: Value, sin: Value): Value {
    x = this.attn.forward(x, cos, sin);
    x = this.mlp.forwardGather(x);
    return x;
  }
}

export interface ModelDims {
  blocks: number;
  experts: number;
  vocab: number;
  logits: number;
  weightFormat: MoeDims['weightFormat'];
  groupSize?: number;
}

export class EagerModel extends nn.Module<[readonly number[]], Value> {
  readonly embedding: nn.Embedding;
  readonly block: nn.ModuleList<EagerBlock>;
  readonly norm = new nn.RMSNorm(HIDDEN, RMS_EPS);
  readonly unembedding: nn.Linear;

  constructor(readonly dims: ModelDims) {
    super();
    this.embedding = new nn.Embedding(dims.vocab, HIDDEN, { device: 'cpu' });
    this.block = new nn.ModuleList(
      Array.from(
        { length: dims.blocks },
        () =>
          new EagerBlock({
            experts: dims.experts,
            hidden: HIDDEN,
            weightFormat: dims.weightFormat,
            groupSize: dims.groupSize,
          }),
      ),
    );
    this.unembedding = new nn.Linear(HIDDEN, dims.logits, { bias: false });
  }

  override forward(tokenIds: readonly number[]): Value {
    const T = tokenIds.length;
    const { cosE, sinE } = ropeConstants(T);
    const cos = tensor(cosE, [T, HEAD]);
    const sin = tensor(sinE, [T, HEAD]);
    let x = this.embedding.forward(tokenIds);
    for (const b of this.block.items) x = b.forward(x, cos, sin);
    return this.unembedding.forward(this.norm.forward(x));
  }
}
