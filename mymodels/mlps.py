from myutils.mi_plugin import BaseBRDF
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)
        freq_bands = freq_bands * self.kwargs.get('frequency_scale', 1.0)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim


    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, input_dims, frequency_scale=1.0):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
        'frequency_scale': frequency_scale,
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

class PositionalEncoding(nn.Module):
    def __init__(self, L):
        """ L: number of frequency bands """
        super(PositionalEncoding, self).__init__()
        self.L= L
        
    def forward(self, inputs):
        L = self.L
        encoded = [inputs]
        for l in range(L):
            encoded.append(torch.sin((2 ** l * math.pi) * inputs))
            encoded.append(torch.cos((2 ** l * math.pi) * inputs))
        return torch.cat(encoded, -1)
class SineLayer(nn.Module):
    ''' Siren layer '''
    
    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias=True, 
                 is_first=False, 
                 omega_0=30, 
                 weight_norm=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # self.init_weights()

        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)

    def init_weights(self):
        if self.is_first:
            nn.init.uniform_(self.linear.weight, 
                             -1 / self.in_features * self.omega_0, 
                             1 / self.in_features * self.omega_0)
        else:
            nn.init.uniform_(self.linear.weight, 
                             -np.sqrt(3 / self.in_features), 
                             np.sqrt(3 / self.in_features))
        nn.init.zeros_(self.linear.bias)

    def forward(self, input):
        return torch.sin(self.linear(input))
class SmoothClamp_real(nn.Module):
    def __init__(self, min_val=0., max_val=1, alpha=5.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.alpha = alpha 

    def forward(self, x):
        lower = self.min_val + (x - self.min_val) * torch.sigmoid(self.alpha * (x - self.min_val))
        upper = self.max_val - (self.max_val - x) * torch.sigmoid(self.alpha * (self.max_val - x))
        y = torch.where(x < self.min_val, lower, x)
        y = torch.where(x > self.max_val, upper, y)
        return y
class SmoothClamp(nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0, alpha=5.0):
        super(SmoothClamp, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.alpha = alpha 

    def forward(self, x):
        lower = self.min_val + (x - self.min_val) * torch.sigmoid(self.alpha * (x - self.min_val))
        upper = self.max_val - (self.max_val - x) * torch.sigmoid(self.alpha * (self.max_val - x))
        return torch.min(torch.max(x, lower), upper)

class PosMLP(BaseBRDF):

    def __init__(self,
                 in_dims,
                 out_dims,
                 dims,
                 skip_connection=(),
                 weight_norm=True,
                 multires_view=0,
                 output_type='envmap',
                 color_ch=5,
                 img_h=None,
                 img_w=None,
                 coordinate_type='uv',
                 normalize_uv=False,
                 use_ste_clamp=True):
        super().__init__()
        self.init_range = np.sqrt(3 / dims[0])
        self.img_h = img_h
        self.img_w = img_w
        self.coordinate_type = coordinate_type
        self.normalize_uv = normalize_uv
        self.use_ste_clamp = use_ste_clamp

        dims = [in_dims] + dims + [out_dims]
        first_omega = 1
        hidden_omega = 1
        self.output_type = output_type

        self.embedview_fn = lambda x: x

        if multires_view > 0:
            embed_dims = 3 if coordinate_type == 'spherical' else 2
            frequency_scale = (
                math.pi
                if coordinate_type == 'uv' and normalize_uv
                else 1.0
            )
            embedview_fn, input_ch = get_embedder(
                multires_view,
                input_dims=embed_dims,
                frequency_scale=frequency_scale,
            )
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - in_dims) + color_ch
        self.num_layers = len(dims)
        self.skip_connection = skip_connection

        for l in range(0, self.num_layers - 1):

            if l + 1 in self.skip_connection:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            is_first = (l == 0) and (multires_view == 0)
            is_last = (l == (self.num_layers - 2))

            if not is_last:
                omega_0 = first_omega if is_first else hidden_omega
                lin = SineLayer(dims[l], out_dim, True, is_first, omega_0,
                                weight_norm)
            else:
                lin = nn.Linear(dims[l], out_dim)
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
                if weight_norm:
                    lin = nn.utils.weight_norm(lin)
                    if torch.isnan(lin.weight).any():
                        raise ValueError(f'nan value in lin{l}.weight')

            setattr(self, "lin" + str(l), lin)

            # self.last_active_fun = nn.Tanh()
            # self.last_active_fun = nn.Identity()
            self.last_active_fun = nn.Softplus()
            # self.last_active_fun = nn.ReLU()
        pass

    def img2points(self, img):
        # img: N, C where N is either a square texture or a 2:1 envmap.
        if self.img_h is not None and self.img_w is not None:
            h = int(self.img_h)
            w = int(self.img_w)
        elif img.shape[0] > 512:
            h = int(img.shape[0] ** 0.5)
            w = h
        else:
            h_float = (img.shape[0] / 2) ** 0.5
            if not h_float.is_integer():
                raise ValueError('width should be double of height')
            h = int(h_float)
            w = h * 2
        x_coords, y_coords = torch.meshgrid(
            torch.arange(h, device=img.device),
            torch.arange(w, device=img.device),
            indexing='ij'
        )

        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()

        if self.coordinate_type == 'spherical':
            u = (y_coords.float() + 0.5) / w
            v = (x_coords.float() + 0.5) / h
            theta = v * np.pi
            phi = u * 2 * np.pi
            sin_theta = torch.sin(theta)
            points = torch.stack([
                sin_theta * torch.cos(phi),
                sin_theta * torch.sin(phi),
                torch.cos(theta),
            ], dim=1)
        else:
            if self.normalize_uv:
                x_coords = 2.0 * (x_coords.float() + 0.5) / h - 1.0
                y_coords = 2.0 * (y_coords.float() + 0.5) / w - 1.0
            points = torch.stack([x_coords, y_coords], dim=1).float()
        embed_points = self.embedview_fn(points)
        points_w_color = torch.cat([embed_points, img], dim=1)
        return points_w_color

    def apply_arm_residual(self, residual, img):
        if self.use_ste_clamp:
            value = 1.3 * nn.Tanh()(residual) + img
            return value.clamp(0,1).detach() + value - value.detach()

        # A logit-space residual stays smoothly bounded without hard-clamp
        # dead zones. The scale matches the old residual's local slope at 0.5.
        base = img.clamp(1e-2, 1.0 - 1e-2)
        return torch.sigmoid(torch.logit(base) + 5.2 * residual)

    @property
    def output_layer(self):
        return getattr(self, "lin" + str(self.num_layers - 2))

    def forward(self, img):
        points = self.img2points(img)
        # breakpoint()
        x = points

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if hasattr(lin, 'linear') and torch.isnan(lin.linear.weight).any():
                raise ValueError(f'nan value in lin{l}.weight')
            elif hasattr(lin, 'weight') and torch.isnan(lin.weight).any():
                raise ValueError(f'nan value in lin{l}.weight')

            if l in self.skip_connection:
                x = torch.cat([x, points], -1)

            x = lin(x)

            if torch.isnan(x).any():
                raise ValueError(f'nan value in x in lin{l}')
        if self.output_type == 'envmap':
            x = nn.Softplus()(x) # make sure positive
        elif self.output_type == 'arm':
            x = self.apply_arm_residual(x, img)
            
        elif self.output_type == 'armn':
            arm = self.apply_arm_residual(x[..., 0:5], img[...,0:5])
            
            normal = x[..., 5:8]
            normal = nn.Tanh()(normal+img[...,5:8])
            out = torch.cat([arm,normal],dim=-1)
            return out
        elif self.output_type == 'normal':
            x = x + img
            x = nn.Tanh()(x)
            x = F.normalize(x, p=2, dim=-1)
        else:
            raise ValueError('output_type should be envmap or arm or armn')
        return x


class NeRFMLP(PosMLP):
    """Standard NeRF-style ReLU backbone with Materialist output heads.

    The coordinate embedding, spherical/UV conventions, and bounded ARM
    residual are deliberately shared with :class:`PosMLP`. This isolates the
    backbone comparison: the canonical configuration uses eight 256-wide
    ReLU hidden layers and one input skip after hidden layer four.
    """

    def __init__(self,
                 in_dims,
                 out_dims,
                 dims,
                 skip_connection=(4,),
                 weight_norm=False,
                 multires_view=0,
                 output_type='envmap',
                 color_ch=5,
                 img_h=None,
                 img_w=None,
                 coordinate_type='uv',
                 normalize_uv=False,
                 use_ste_clamp=True):
        BaseBRDF.__init__(self)
        if not dims:
            raise ValueError('NeRFMLP needs at least one hidden layer')
        self.img_h = img_h
        self.img_w = img_w
        self.coordinate_type = coordinate_type
        self.normalize_uv = normalize_uv
        self.use_ste_clamp = use_ste_clamp
        self.output_type = output_type
        self.skip_connection = tuple(int(index) for index in skip_connection)
        self.embedview_fn = lambda x: x

        feature_dims = in_dims
        if multires_view > 0:
            embed_dims = 3 if coordinate_type == 'spherical' else 2
            frequency_scale = (
                math.pi
                if coordinate_type == 'uv' and normalize_uv
                else 1.0
            )
            embedview_fn, embedded_dims = get_embedder(
                multires_view,
                input_dims=embed_dims,
                frequency_scale=frequency_scale,
            )
            self.embedview_fn = embedview_fn
            feature_dims = embedded_dims + color_ch

        self.feature_dims = feature_dims
        self.hidden_layers = nn.ModuleList()
        previous_dims = feature_dims
        for layer_index, hidden_dims in enumerate(dims):
            if layer_index > 0 and (layer_index - 1) in self.skip_connection:
                previous_dims += feature_dims
            layer = nn.Linear(previous_dims, hidden_dims)
            if weight_norm:
                layer = nn.utils.weight_norm(layer)
            self.hidden_layers.append(layer)
            previous_dims = hidden_dims
        self._output_layer = nn.Linear(previous_dims, out_dims)
        nn.init.zeros_(self._output_layer.weight)
        nn.init.zeros_(self._output_layer.bias)

    @property
    def output_layer(self):
        return self._output_layer

    def forward(self, img):
        points = self.img2points(img)
        x = points
        for layer_index, layer in enumerate(self.hidden_layers):
            x = F.relu(layer(x))
            if (
                layer_index in self.skip_connection
                and layer_index + 1 < len(self.hidden_layers)
            ):
                x = torch.cat([x, points], dim=-1)
        x = self._output_layer(x)

        if self.output_type == 'envmap':
            return F.softplus(x)
        if self.output_type == 'arm':
            return self.apply_arm_residual(x, img)
        if self.output_type == 'armn':
            arm = self.apply_arm_residual(x[..., 0:5], img[..., 0:5])
            normal = torch.tanh(x[..., 5:8] + img[..., 5:8])
            return torch.cat([arm, normal], dim=-1)
        if self.output_type == 'normal':
            return F.normalize(torch.tanh(x + img), p=2, dim=-1)
        raise ValueError('output_type should be envmap or arm or armn')
