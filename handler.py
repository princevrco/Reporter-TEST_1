from option_c.service import authenticated_handler

if __name__=='__main__':
    import runpod
    runpod.serverless.start({'handler':authenticated_handler})
