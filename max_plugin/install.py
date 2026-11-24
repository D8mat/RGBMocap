"""Run in 3ds Max to install the RGB MoCap panel for the current user."""
from pathlib import Path
import shutil,datetime
from pymxs import runtime as rt

def install():
    source=Path(__file__).resolve().parent
    target=Path(str(rt.getDir(rt.name('userScripts'))))/'RGBMoCap'
    macro=Path(str(rt.getDir(rt.name('userMacros'))))/'RGBMoCap-FBXLiveLink.mcr'
    names=('rgbmocap_max.py','rgbmocap_max_protocol.py','rgbmocap_max_scene.py','README.md','LICENSE')
    target.mkdir(parents=True,exist_ok=True)
    existing=[target/n for n in names if (target/n).exists()]
    if existing or macro.exists():
        backup=target/('backup-'+datetime.datetime.now().strftime('%Y%m%d-%H%M%S'));backup.mkdir()
        for p in existing:shutil.copy2(p,backup/p.name)
        if macro.exists():shutil.copy2(macro,backup/macro.name)
    for n in names:shutil.copy2(source/n,target/n)
    path=str(target/'rgbmocap_max.py').replace('"','""')
    macro.write_text('macroScript RGBMoCapFBXLiveLink category:"RGB MoCap" tooltip:"RGB MoCap FBX Live Link"\n(\n on execute do python.executeFile @"'+path+'"\n)\n',encoding='utf-8')
    rt.fileIn(str(macro))
    print('RGB MoCap installed: '+str(target))
    return str(target)

if __name__=='__main__':install()
