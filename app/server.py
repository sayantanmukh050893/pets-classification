import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/u/0/uc?id=1-Nc2rdnc4nQQhfm9-e1c5xqboSTqDNLj&export=download'
export_file_name = 'export.pkl'

classes = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']
#classes = ['Animal : Cat, Breed : Abyssinian', 'Animal : Cat, Breed : Bengal', 'Animal : Cat, Breed: Birman', 'Animal : Cat, Breed : Bombay', 'Animal : Cat, Breed : British Shorthair', 'Animal : Cat, Breed : Egyptian Mau', 'Animal : Cat, Breed : Maine Coon', 'Animal : Cat, Breed : Persian', 'Animal : Cat, Breed : Ragdoll', 'Animal : Cat, Breed : Russian Blue', 'Animal : Cat, Breed : Siamese', 'Animal : Cat, Breed : Sphynx', 'Animal : Dog, Breed : American Bulldog', 'Animal : Dog, Breed : American Pit Bull Terrier', 'Animal : Dog, Breed : Basset Hound', 'Animal : Dog, Breed : Beagle', 'Animal : Dog, Breed : Boxer', 'Animal : Dog, Breed : Chihuahua', 'Animal : Dog, Breed : English Cocker Spaniel', 'Animal : Dog, Breed : English Setter', 'Animal : Dog, Breed : German Shorthaired', 'Animal : Dog, Breed : Great Pyrenees', 'Animal : Dog, Breed : Havanese', 'Animal : Dog, Breed : Japanese Chin', 'Animal : Dog, Breed : Keeshond', 'Animal : Dog, Breed : Leonberger', 'Animal : Dog, Breed : Miniature Pinscher', 'Animal : Dog, Breed : Newfoundland', 'Animal : Dog, Breed : Pomeranian', 'Animal : Dog, Breed : Pug', 'Animal : Dog, Breed : Saint Bernard', 'Animal : Dog, Breed : Samoyed', 'Animal : Dog, Breed : Scottish Terrier', 'Animal : Dog, Breed : Shiba Inu', 'Animal : Dog, Breed : Staffordshire Bull Terrier', 'Animal : Dog, Breed : Wheaten Terrier', 'Animal : Dog, Breed : Yorkshire Terrier']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    cats = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx']
    dogs = ['american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier']
    animal = None
    #if(str(prediction) in cats):
     #   animal = "cat"
    #elif(str(prediction) in dogs):
     #   animal = "dog"
    #print(animal) 
    return JSONResponse({'The above image is of a ',str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
