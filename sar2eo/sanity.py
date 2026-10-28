import argparse, torch
try:
    from .model import SAR2EO
except ImportError:
    from sar2eo.model import SAR2EO

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--debug", action="store_true"); ap.add_argument("--sar-checkpoint", default="assets/sar_jepa/checkpoint-200.pth"); a = ap.parse_args()
    d, depth, heads, k = (384, 12, 6, 4) if a.debug else (768, 12, 12, 4)
    m = SAR2EO(d, depth, heads, k); 
    if not a.debug:
        try:
            from .sar_encoder import load_sar_checkpoint
        except ImportError:
            from sar2eo.sar_encoder import load_sar_checkpoint
        load_sar_checkpoint(m.sar_encoder, a.sar_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"; m.to(device)
    sar, x, t = torch.randn(2,1,256,256,device=device), torch.randn(2,4,32,32,device=device), torch.rand(2,device=device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
        v,z = m(x,t,sar,torch.zeros(2,device=device,dtype=torch.bool),sar_trainable=True)
        loss = v.float().square().mean() + z.float().square().mean()
    assert sar.shape == (2,1,256,256); assert m.sar_encoder(sar).shape == (2,256,768)
    assert m.encode_sar(sar)[0].shape == (2,256,d); assert m.encode_sar(sar)[1].shape == (2,d)
    assert x.shape == (2,4,32,32) and v.shape == x.shape and z.shape == (2,256,768)
    assert torch.isfinite(loss); loss.backward()
    print({"device":device,"v":tuple(v.shape),"z":tuple(z.shape),"parameters":sum(p.numel() for p in m.parameters()), "sar_parameters":sum(p.numel() for p in m.sar_encoder.parameters()), "generator_parameters":sum(p.numel() for p in m.parameters())-sum(p.numel() for p in m.sar_encoder.parameters())})

if __name__ == "__main__": main()
