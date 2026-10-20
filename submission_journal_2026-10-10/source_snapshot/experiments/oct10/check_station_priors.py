"""Verify a customer's own previous label cannot change its peer-prior features."""
import numpy as np
import pandas as pd
from run_station_context import peer_priors
import feature_pipeline as fp


def main():
    _,_,_,classes=fp.load_config()
    columns=["prev__"+fp.slug(c) for c in classes]
    frame=pd.DataFrame(0.0,index=["a","b","c","d"],columns=columns)
    frame.loc["a","prev__"+fp.slug("Inactivity")]=1
    frame.loc["b","prev__"+fp.slug("Stable")]=1
    frame.loc["c","prev__"+fp.slug("Existing-category growth: Fuel")]=1
    # Customer d has no prior label and a station with no known peers.
    sites=pd.Series([10,10,20,30],index=frame.index)
    original=pd.DataFrame(peer_priors(frame,sites,classes),index=frame.index)
    changed=frame.copy()
    changed.loc["a",columns]=0
    changed.loc["a","prev__"+fp.slug("Existing-category growth: Fuel")]=1
    revised=pd.DataFrame(peer_priors(changed,sites,classes),index=frame.index)
    np.testing.assert_allclose(original.loc["a"],revised.loc["a"],rtol=0,atol=1e-14)
    assert not np.allclose(original.loc["b"],revised.loc["b"])
    assert np.isfinite(original.to_numpy()).all()
    priors=original.filter(regex="station_prior_(?!peers)")
    np.testing.assert_allclose(priors.sum(axis=1),1,rtol=1e-12,atol=1e-12)
    assert original.loc["c","station_prior_peers"]==0
    assert original.loc["d","station_prior_peers"]==0
    print("Passed self-exclusion, peer sensitivity, singleton/unknown-prior and finite-probability checks.")


if __name__=="__main__":main()
