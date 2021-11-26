# Class to save and show result neatly

class ResultTable:
    def __init__(self, header=None, splitter='||', int_formatter='%3d', float_formatter='%.4f'):
        self.header = header
        if self.header is not None:
            self.set_headers(self.header)
        self.num_rows = 0
        self.splitter = splitter
        self.int_formatter = int_formatter
        self.float_formatter = float_formatter

    def set_headers(self, header):
        self.header = ['NAME'] + header
        self.data = {h: [] for h in self.header}
        self.max_len = {h: len(h) for h in self.header}

    def add_row(self, row_name, row_dict):
        # If header is not defined, fetch from input dict
        if self.header is None:
            self.set_headers(list(row_dict.keys()))

        # If input dict has new column, make one
        for key in row_dict:
            if key not in self.data:
                self.data[key] = ['-'] * self.num_rows
                self.header.append(key)

        for h in self.header:
            if h == 'NAME':
                self.data['NAME'].append(row_name)
                self.max_len[h] = max(self.max_len['NAME'], len(row_name))
            else:
                # If input dict doesn't have values for table header, make empty value.
                if h not in row_dict:
                    row_dict[h] = '-'

                # convert input dict to string
                d = row_dict[h]
                if isinstance(d, int):
                    d_str = self.int_formatter % d
                elif isinstance(d, float):
                    d_str = self.float_formatter % d
                elif isinstance(d, str):
                    d_str = d
                else:
                    raise NotImplementedError

                self.data[h].append(d_str)
                self.max_len[h] = max(self.max_len[h], len(d_str))
        self.num_rows += 1

    def row_to_line(self, row_values):
        value_str = []
        for i, max_length in enumerate(self.max_len.values()):
            length = len(row_values[i])
            diff = max_length - length
            # # Center align
            # left_space = diff // 2
            # right_space = diff - left_space
            # s = ' ' * left_space + row_values[i] + ' ' * right_space

            # Left align
            s = row_values[i] + ' ' * diff
            value_str.append(s)

        return self.splitter + ' ' + (' %s ' % self.splitter).join(value_str) + ' ' + self.splitter

    def to_string(self):
        size_per_col = {h: self.max_len[h] + 2 + len(self.splitter) for h in self.max_len}
        line_len = sum([size_per_col[c] for c in size_per_col]) + len(self.splitter)
        table_str = '\n'

        # HEADER
        line = self.row_to_line(self.header)
        table_str += '=' * line_len + '\n'
        table_str += line + '\n'
        table_str += self.splitter + '-' * (line_len - len(self.splitter) * 2) + self.splitter + '\n'
        # DATA
        for row_values in zip(*self.data.values()):
            line = self.row_to_line(row_values)
            table_str += line + '\n'
        table_str += '=' * line_len + '\n'
        return table_str

    def show(self):
        print(self.to_string())

    @property
    def shape(self):
        return (self.num_rows, self.num_cols)

    @property
    def num_cols(self):
        return len(self.header)