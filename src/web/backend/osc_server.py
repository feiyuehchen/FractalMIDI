from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
import asyncio
import logging
import json

logger = logging.getLogger(__name__)

class OSCBridge:
    def __init__(self, ip="127.0.0.1", port=5005, inference_engine=None):
        self.ip = ip
        self.port = port
        self.inference_engine = inference_engine
        self.dispatcher = Dispatcher()
        self.transport = None
        self.protocol = None
        
        # Register handlers
        self.dispatcher.map("/fractal/param/*", self.handle_param)
        self.dispatcher.map("/fractal/generate", self.handle_generate)
        self.dispatcher.map("/fractal/latent/*", self.handle_latent)
        
    def handle_param(self, address, *args):
        """Handle parameter updates (temp, cfg, etc)."""
        param_name = address.split("/")[-1]
        val = args[0] if len(args) == 1 else list(args)
        logger.info(f"OSC Param: {param_name} = {val}")
        
        if self.inference_engine:
            if not hasattr(self.inference_engine, 'osc_state'):
                self.inference_engine.osc_state = {}
            self.inference_engine.osc_state[param_name] = val

    def handle_latent(self, address, *args):
        """Handle latent vector updates."""
        # /fractal/latent/level0  0.1 0.2 ...
        layer_name = address.split("/")[-1]
        vector = list(args)
        logger.info(f"OSC Latent {layer_name} update: len={len(vector)}")
        
        if self.inference_engine:
            if not hasattr(self.inference_engine, 'osc_latents'):
                self.inference_engine.osc_latents = {}
            self.inference_engine.osc_latents[layer_name] = vector

    def handle_generate(self, address, *args):
        """Trigger generation."""
        logger.info("OSC Trigger Generation")
        if self.inference_engine:
            # We need to schedule this on the loop
            loop = asyncio.get_event_loop()
            # Construct a default request using OSC params
            from src.web.backend.app import GenerationRequest
            
            state = getattr(self.inference_engine, 'osc_state', {})
            
            try:
                req = GenerationRequest(
                    mode=state.get('mode', 'unconditional'),
                    length=int(state.get('length', 256)),
                    temperature=float(state.get('temperature', 1.0)),
                    cfg=float(state.get('cfg', 1.0))
                )
                
                # Create a job ID
                import uuid
                job_id = f"osc_{uuid.uuid4()}"
                
                # Run in background
                loop.create_task(self.inference_engine.generate(job_id, req))
                
            except Exception as e:
                logger.error(f"OSC Gen Error: {e}")

    async def start(self):
        """Start the OSC server."""
        loop = asyncio.get_event_loop()
        # AsyncIOOSCUDPServer logic slightly different for manual start
        # We use loop.create_datagram_endpoint directly with a protocol factory
        
        # Wrapper to adapt python-osc to asyncio protocol
        class _OSCProtocol(asyncio.DatagramProtocol):
            def __init__(self, dispatcher):
                self.dispatcher = dispatcher
            def connection_made(self, transport):
                self.transport = transport
            def datagram_received(self, data, addr):
                self.dispatcher.call_handlers_for_packet(data, addr)

        try:
            self.transport, self.protocol = await loop.create_datagram_endpoint(
                lambda: _OSCProtocol(self.dispatcher),
                local_addr=(self.ip, self.port)
            )
            logger.info(f"OSC Server listening on {self.ip}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start OSC server: {e}")

    def stop(self):
        if self.transport:
            self.transport.close()
