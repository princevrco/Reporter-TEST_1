from .data import check_cfg,frame,window,records,stamp


def scenario(cfg,start,end,temperature_c,covariates=None):
    check_cfg(cfg)
    seed=[{'timestamp':stamp(start,cfg['timezone'],True).isoformat(),'quality':'good','temperature_c':temperature_c}]
    f,_,_=window(frame(seed,cfg),cfg,start,end)
    f['temperature_c']=float(temperature_c);f['quality']='good';f['meter_id']=cfg['meter_id'];f['conditions_source']='user_assumed_weather_scenario'
    values=covariates or {}
    for col in cfg.get('independent_variables',[]):
        if col not in values:raise ValueError('Supply every independent-variable scenario value')
        f[col]=float(values[col])
    return records(f)
